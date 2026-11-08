"""Base class and shared types for all MediBot specialist agents."""
import logging
from typing import Optional

from pydantic import Field

from src.llm_manager import get_chat_model, LLMError

logger = logging.getLogger(__name__)


class AgentResult:
    """Structured result returned by every specialist agent."""

    def __init__(self, agent_name: str, response: str, confidence: float = 1.0,
                 grounded: bool = False, sources: Optional[list] = None, error: Optional[str] = None):
        self.agent_name = agent_name
        self.response = response
        self.confidence = confidence
        self.grounded = grounded
        self.sources = sources or []
        self.error = error

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "response": self.response,
            "confidence": self.confidence,
            "grounded": self.grounded,
            "sources": self.sources,
            "error": self.error,
        }


class BaseAgent:
    """Common functionality: role-bound failover LLM and prompt invocation."""

    name: str = "base"
    role: str = "medical"  # key into ROLE_MODELS / ROLE_LLM_CHAINS

    def __init__(self, **kwargs):
        # Allow tests to inject a temperature/max_tokens override.
        self._model_kwargs = kwargs

    @property
    def llm(self):
        """Failover chat model bound to this agent's role."""
        return get_chat_model(self.role)

    def invoke(self, prompt: str) -> str:
        """Invoke the role's failover LLM with a plain prompt."""
        from langchain_core.messages import HumanMessage
        from src.llm_manager import LLMError

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.error("LLM invoke failed for role=%s: %s", self.role, exc)
            raise LLMError(f"LLM invoke failed for role '{self.role}': {exc}") from exc

        # ---------- DIAGNOSTIC (temporary) ----------
        try:
            logger.warning("=== LLM RAW RESULT role=%s ===", self.role)
            logger.warning("type=%s", type(result).__name__)
            logger.warning("repr=%.800r", result)
            logger.warning("content=%r", getattr(result, "content", "<no content attr>"))
            logger.warning("additional_kwargs=%r", getattr(result, "additional_kwargs", None))
            logger.warning("response_metadata=%r", getattr(result, "response_metadata", None))
            logger.warning("usage_metadata=%r", getattr(result, "usage_metadata", None))
            logger.warning("=== END LLM RAW RESULT ===")
        except Exception as _diag_exc:
            logger.warning("Diagnostic logging failed: %s", _diag_exc)
        # -------------------------------------------

        content = getattr(result, "content", None)
        if content is None:
            content = str(result) if result is not None else ""
        if isinstance(content, list):
            # Some providers return content as a list of blocks
            content = "".join(
                (b.get("text", "") if isinstance(b, dict) else str(b)) for b in content
            )
        if not content or not content.strip():
            raise LLMError(f"LLM returned an empty completion for role '{self.role}'")
        return content