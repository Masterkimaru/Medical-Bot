"""LLM configuration and provider management with automatic failover.

Design:
- Each agent role has a provider chain (e.g. openrouter -> groq) from config.
- Errors are classified: retryable (timeouts, 5xx, rate limits) vs auth (401/403)
  vs invalid request (other 4xx) vs application-level errors.
- Only provider-layer failures trigger fallback to the next provider in the
  chain; every event is logged so failures are auditable.
- Delegates (OpenRouterChatLLM / ChatGroq) are cached and reused per process.

Empty-completion handling (added):
- Some OpenRouter routes return HTTP 200 with an empty "content" and a
  "finish_reason" of "content_filter" or similar. The previous code turned
  that into a silent AIMessage(content='') which downstream looked like a
  successful but empty response. We now detect that case explicitly and raise
  LLMUnavailableError so the failover wrapper moves to the next provider,
  and so the caller gets a clear error instead of an empty draft.
- Top-level "error" fields that some OpenRouter proxies return with HTTP 200
  are also detected and surfaced.
- response_metadata / usage_metadata are populated so diagnostic logging in
  BaseAgent.invoke() can see finish_reason / usage directly.
"""
import logging
import threading
import time
from typing import Any, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MEDICAL_FALLBACK_LLM_MODEL,
    MEDICAL_LLM_MODEL,
    MENTAL_HEALTH_FALLBACK_LLM_MODEL,
    MENTAL_HEALTH_LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ROLE_LLM_CHAINS,
    SITE_NAME,
    SITE_URL,
    VISION_LLM_MODEL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM-layer failures."""


class LLMAuthenticationError(LLMError):
    """Provider rejected credentials (401/403). Not fixed by retrying."""


class LLMInvalidRequestError(LLMError):
    """Provider rejected the request itself (400/404/422, e.g. unknown model)."""


class LLMUnavailableError(LLMError):
    """Temporary provider problem: timeout, connection error, 429, 5xx,
    or a successful response with an empty completion (content filter, etc.)."""


class LLMAllProvidersFailedError(LLMError):
    """Every provider in the role's chain failed."""


def classify_llm_error(exc: Exception) -> str:
    """Classify an LLM exception: 'auth' | 'rate_limit' | 'invalid' | 'retryable' | 'app'."""
    if isinstance(exc, LLMAuthenticationError):
        return "auth"
    if isinstance(exc, LLMInvalidRequestError):
        return "invalid"
    if isinstance(exc, (LLMUnavailableError, LLMAllProvidersFailedError)):
        return "retryable"
    status = getattr(exc, "status_code", None)
    if status is not None:
        if status in (401, 403):
            return "auth"
        if status == 429:
            return "rate_limit"
        if 400 <= status < 500:
            return "invalid"
        return "retryable"
    name = type(exc).__name__
    if "Auth" in name or "Permission" in name:
        return "auth"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "retryable"
    return "app"


# --------------------------------------------------------------------------
# OpenRouter chat model (LangChain BaseChatModel implementation)
# --------------------------------------------------------------------------
_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "user", "function": "user"}


def _extract_text_content(content: Any) -> str:
    """Normalise OpenRouter 'message.content' which may be a string or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # Common OpenRouter/OpenAI content-block shapes:
                if "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
                elif block.get("type") == "text" and isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class OpenRouterChatLLM(BaseChatModel):
    """Chat model that talks to the OpenRouter chat-completions API."""

    model_name: str = Field(..., description="OpenRouter model identifier")
    temperature: float = Field(LLM_TEMPERATURE)
    max_tokens: int = Field(LLM_MAX_TOKENS)
    timeout: float = Field(LLM_TIMEOUT)

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not OPENROUTER_API_KEY:
            raise LLMAuthenticationError("OPENROUTER_API_KEY is not configured.")

        if self.max_tokens is None or int(self.max_tokens) <= 0:
            # A zero/negative max_tokens is a common misconfig that produces
            # empty completions from every provider. Fail loudly instead.
            raise LLMInvalidRequestError(
                f"LLM_MAX_TOKENS is misconfigured (value={self.max_tokens!r}); it must be > 0."
            )

        payload_messages = [{"role": _ROLE_MAP.get(m.type, "user"), "content": m.content} for m in messages]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if SITE_URL:
            headers["HTTP-Referer"] = SITE_URL
        if SITE_NAME:
            headers["X-OpenRouter-Title"] = SITE_NAME

        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        last_error: Optional[Exception] = None
        attempts = max(1, LLM_MAX_RETRIES + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers, json=payload, timeout=self.timeout,
                )
                if response.status_code in (401, 403):
                    raise LLMAuthenticationError(
                        f"OpenRouter rejected credentials (HTTP {response.status_code})."
                    )
                if response.status_code == 429:
                    last_error = LLMUnavailableError(f"OpenRouter rate limited model {self.model_name}.")
                    wait = min(2 ** attempt, 8)
                    logger.warning("OpenRouter 429; retrying in %ss (attempt %d/%d).", wait, attempt, attempts)
                    time.sleep(wait)
                    continue
                if response.status_code in (400, 404, 422):
                    raise LLMInvalidRequestError(
                        f"OpenRouter rejected request for {self.model_name} "
                        f"(HTTP {response.status_code}): {response.text[:300]}"
                    )
                response.raise_for_status()
                data = response.json()

                # Some OpenRouter proxies return HTTP 200 with a top-level "error"
                # when an upstream provider dies mid-stream. Detect and surface it.
                if isinstance(data, dict) and data.get("error"):
                    err = data["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    logger.error(
                        "[openrouter] top-level error for %s: %r (raw=%r)",
                        self.model_name, err, data,
                    )
                    # Treat as retryable so the failover wrapper can try the next provider.
                    last_error = LLMUnavailableError(f"OpenRouter upstream error: {msg}")
                    if attempt < attempts:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    raise last_error

                choices = (data or {}).get("choices") or []
                if not choices:
                    logger.error(
                        "[openrouter] no choices returned for %s: %r",
                        self.model_name, data,
                    )
                    raise LLMInvalidRequestError(
                        f"OpenRouter returned no choices for {self.model_name}: {str(data)[:300]}"
                    )

                choice = choices[0] or {}
                finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}

                # Some models/providers put a structured refusal into a separate key.
                refusal = message.get("refusal")
                if refusal:
                    logger.error(
                        "[openrouter] refusal from %s (finish_reason=%r): %r",
                        self.model_name, finish_reason, refusal,
                    )
                    raise LLMInvalidRequestError(
                        f"OpenRouter refused the request for {self.model_name} "
                        f"(finish_reason={finish_reason}, refusal={refusal!r})"
                    )

                content = _extract_text_content(message.get("content"))

                if not content or not content.strip():
                    # IMPORTANT: do not silently return AIMessage('') here.
                    # finish_reason="content_filter" is the most common cause;
                    # also seen with "length" when max_tokens is too small.
                    usage = (data or {}).get("usage") or {}
                    logger.error(
                        "[openrouter] empty content from %s: finish_reason=%r usage=%r message=%r",
                        self.model_name, finish_reason, usage, message,
                    )
                    last_error = LLMUnavailableError(
                        f"OpenRouter returned an empty completion for {self.model_name} "
                        f"(finish_reason={finish_reason}). Typical causes: content_filter, "
                        f"max_tokens too small, or the upstream model is unavailable."
                    )
                    # Retry on transient causes; raise immediately on content_filter.
                    if finish_reason == "content_filter":
                        raise last_error
                    if attempt < attempts and finish_reason in (None, "length"):
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    raise last_error

                usage = (data or {}).get("usage") or {}
                ai_msg = AIMessage(
                    content=content,
                    response_metadata={
                        "provider": "openrouter",
                        "model": self.model_name,
                        "finish_reason": finish_reason,
                        "id": (data or {}).get("id"),
                        "usage": usage,
                    },
                    usage_metadata=(
                        {
                            "input_tokens": usage.get("prompt_tokens", 0) or 0,
                            "output_tokens": usage.get("completion_tokens", 0) or 0,
                            "total_tokens": usage.get("total_tokens", 0) or 0,
                        }
                        if usage else None
                    ),
                )
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])

            except LLMError:
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = LLMUnavailableError(f"OpenRouter network error: {exc}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))
            except requests.exceptions.RequestException as exc:
                last_error = LLMUnavailableError(f"OpenRouter request failed: {exc}")
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMInvalidRequestError(f"Unexpected OpenRouter response shape: {exc}") from exc
        raise last_error or LLMUnavailableError("OpenRouter request failed for unknown reasons.")


# --------------------------------------------------------------------------
# Failover wrapper
# --------------------------------------------------------------------------
_DELEGATES = {}
_delegate_lock = threading.Lock()


def _build_spec(provider: str, model: str, temperature: float, max_tokens: int) -> dict:
    return {"provider": provider, "model": model, "temperature": temperature, "max_tokens": max_tokens}


def _get_delegate(spec: dict):
    """Get or build a cached delegate for a provider/model spec."""
    key = (spec["provider"], spec["model"], spec["temperature"], spec["max_tokens"])
    if key in _DELEGATES:
        return _DELEGATES[key]
    with _delegate_lock:
        if key not in _DELEGATES:
            if spec["provider"] == "openrouter":
                delegate: Any = OpenRouterChatLLM(
                    model_name=spec["model"],
                    temperature=spec["temperature"],
                    max_tokens=spec["max_tokens"],
                    timeout=LLM_TIMEOUT,
                )
            elif spec["provider"] == "groq":
                from langchain_groq import ChatGroq  # imported lazily

                if not GROQ_API_KEY:
                    raise LLMAuthenticationError("GROQ_API_KEY is not configured.")
                delegate = ChatGroq(
                    model=spec["model"],
                    api_key=GROQ_API_KEY,
                    temperature=spec["temperature"],
                    max_tokens=spec["max_tokens"],
                    timeout=LLM_TIMEOUT,
                )
            else:
                raise LLMInvalidRequestError(f"Unknown LLM provider: {spec['provider']}")
            _DELEGATES[key] = delegate
    return _DELEGATES[key]


class FailoverChatModel(BaseChatModel):
    """A chat model that walks a provider chain until one succeeds."""

    role: str = Field(..., description="Agent role this model serves (e.g. 'medical')")

    def _specs(self) -> List[dict]:
        providers = ROLE_LLM_CHAINS.get(self.role, ["openrouter"])
        specs: List[dict] = []
        for provider in providers:
            model = self._model_for_provider(provider)
            if model:
                specs.append(_build_spec(provider, model, LLM_TEMPERATURE, LLM_MAX_TOKENS))
        if not specs:
            raise LLMAuthenticationError(
                f"No LLM provider configured for role '{self.role}'. "
                f"Set OPENROUTER_API_KEY (and optionally GROQ_API_KEY)."
            )
        return specs

    def _model_for_provider(self, provider: str) -> Optional[str]:
        primary, fallback = ROLE_MODELS[self.role]
        if provider == "openrouter":
            return primary
        if provider == "groq":
            return fallback
        return None

    @property
    def _llm_type(self) -> str:
        return "failover"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last_error: Optional[Exception] = None
        for spec in self._specs():
            try:
                delegate = _get_delegate(spec)
                return delegate._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except LLMInvalidRequestError as exc:
                # The request itself is broken for this model; log and try the next model.
                logger.error("[%s] invalid request on %s/%s: %s", self.role, spec["provider"], spec["model"], exc)
                last_error = exc
            except LLMAuthenticationError as exc:
                logger.error("[%s] authentication failure on %s: %s", self.role, spec["provider"], exc)
                last_error = exc
            except Exception as exc:
                kind = classify_llm_error(exc)
                logger.warning("[%s] %s failure on %s/%s: %s",
                               self.role, kind, spec["provider"], spec["model"], exc)
                last_error = exc
        raise LLMAllProvidersFailedError(
            f"All LLM providers failed for role '{self.role}': {last_error}"
        ) from last_error


# Model names per role: (openrouter primary, groq fallback)
ROLE_MODELS = {
    "medical": (MEDICAL_LLM_MODEL, MEDICAL_FALLBACK_LLM_MODEL),
    "general_health": (MEDICAL_LLM_MODEL, MEDICAL_FALLBACK_LLM_MODEL),
    "mental_health": (MENTAL_HEALTH_LLM_MODEL, MENTAL_HEALTH_FALLBACK_LLM_MODEL),
    "validation": (MEDICAL_LLM_MODEL, None),
    "vision": (VISION_LLM_MODEL, None),
}

_model_cache = {}
_model_cache_lock = threading.Lock()


def get_chat_model(role: str) -> FailoverChatModel:
    """Return a failover chat model for an agent role."""
    key = role
    if key not in _model_cache:
        with _model_cache_lock:
            if key not in _model_cache:
                _model_cache[key] = FailoverChatModel(role=role)
    return _model_cache[key]


def analyze_image(base64_image: str, prompt: str) -> str:
    """Analyze a base64 image with the configured OpenRouter vision model."""
    if not OPENROUTER_API_KEY:
        raise LLMAuthenticationError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-OpenRouter-Title"] = SITE_NAME

    payload = {
        "model": VISION_LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }],
        "temperature": 0.3,
        "max_tokens": 700,
    }

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=LLM_TIMEOUT,
        )
        if response.status_code in (401, 403):
            raise LLMAuthenticationError("OpenRouter rejected credentials for image analysis.")
        if response.status_code == 429:
            raise LLMUnavailableError("OpenRouter rate limited the vision model.")
        if 400 <= response.status_code < 500:
            raise LLMInvalidRequestError(f"Vision request rejected (HTTP {response.status_code}).")
        response.raise_for_status()
        result = response.json()

        if isinstance(result, dict) and result.get("error"):
            err = result["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise LLMUnavailableError(f"Vision upstream error: {msg}")

        choices = result.get("choices") or []
        if not choices:
            raise LLMInvalidRequestError(
                f"Vision request returned no choices: {str(result)[:300]}"
            )
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        content = _extract_text_content(message.get("content"))
        if not content or not content.strip():
            logger.error(
                "[openrouter-vision] empty content for %s: finish_reason=%r message=%r",
                VISION_LLM_MODEL, finish_reason, message,
            )
            raise LLMUnavailableError(
                f"Vision model returned an empty completion for {VISION_LLM_MODEL} "
                f"(finish_reason={finish_reason})."
            )
        return content

    except LLMError:
        raise
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise LLMUnavailableError(f"Image analysis network error: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailableError(f"Image analysis request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMInvalidRequestError(f"Unexpected image analysis response: {exc}") from exc