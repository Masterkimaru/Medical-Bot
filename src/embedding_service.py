"""Embedding service.

Default provider: local Sentence-Transformers (all-MiniLM-L6-v2, 384 dims).
Optional providers (activate by setting EMBEDDING_PROVIDER in .env):
  - "local"      : SentenceTransformer, 384 dims, index PINECONE_INDEX_LOCAL
  - "openrouter" : nvidia/nemotron-3-embed-1b:free via OpenRouter, 2048 dims
  - "pinecone"   : llama-text-embed-v2 via Pinecone Inference, 1024 dims

Key safety rule: embeddings from different models are NOT interchangeable.
This service tags every embedding request with the provider that generated it,
and the vector service uses that tag to select the matching Pinecone index.

Asymmetric models (NVIDIA nemotron, Pinecone llama-text-embed-v2) REQUIRE
input_type="passage" for documents and input_type="query" for queries.
Omitting this silently degrades retrieval quality.

All returned vectors are coerced to Python float — Pinecone rejects int values.
"""
import logging
import importlib
import threading
import time
from typing import List, Optional

import requests

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROVIDER,
    EMBEDDING_TIMEOUT,
    LOCAL_EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    SITE_NAME,
    SITE_URL,
)

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding failure that is unlikely to be fixed by retrying or fallback."""


class EmbeddingUnavailableError(EmbeddingError):
    """Provider temporarily unavailable (network, 429, 5xx, timeout) - fallback allowed."""


class EmbeddingDimensionMismatch(EmbeddingError):
    """Embedding dimension does not match the configured dimension for the provider."""


# Pinecone hosted model (used only when EMBEDDING_PROVIDER=pinecone)
PINECONE_EMBEDDING_MODEL = "llama-text-embed-v2"

_VALID_PROVIDERS = ("local", "openrouter", "pinecone")


def _to_float_vectors(vectors: List[List[float]]) -> List[List[float]]:
    """Coerce every element to Python float.

    Pinecone's client strictly validates that each value in a vector is a
    float and raises PineconeApiTypeError when it encounters an int. JSON
    parsing can yield ints for whole-number values (e.g. 0 instead of 0.0),
    so we normalise defensively here.
    """
    return [[float(x) for x in vec] for vec in vectors]


class EmbeddingService:
    """Reusable embedding service supporting single and batch embedding."""

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        provider: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        local_model: Optional[str] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        timeout: float = EMBEDDING_TIMEOUT,
    ):
        raw = (provider or EMBEDDING_PROVIDER or "local").lower()
        if raw not in _VALID_PROVIDERS:
            logger.warning(
                "Unknown EMBEDDING_PROVIDER=%r; defaulting to 'local'. Valid values: %s",
                raw, ", ".join(_VALID_PROVIDERS),
            )
            raw = "local"
        self.provider = raw

        self.openrouter_model = openrouter_model or OPENROUTER_EMBEDDING_MODEL
        self.local_model_name = local_model or LOCAL_EMBEDDING_MODEL
        self.batch_size = max(1, batch_size)
        self.timeout = timeout

        self._local_model = None
        self._local_lock = threading.Lock()
        self._session = requests.Session()
        self._pc = None  # lazy Pinecone client

        if self.provider == "local":
            logger.info(
                "EmbeddingService initialised with LOCAL provider (model=%s, dim=%d).",
                self.local_model_name, self.dimension,
            )

    # ------------------------------------------------------------------
    # Provider bookkeeping
    # ------------------------------------------------------------------
    @property
    def provider_tag(self) -> str:
        if self.provider == "openrouter":
            return "openrouter"
        if self.provider == "pinecone":
            return "pinecone"
        return "local"

    @property
    def model_name(self) -> str:
        if self.provider_tag == "openrouter":
            return self.openrouter_model
        if self.provider_tag == "pinecone":
            return PINECONE_EMBEDDING_MODEL
        return self.local_model_name

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSIONS[self.provider_tag]

    def switch_to_pinecone(self) -> None:
        if self.provider_tag != "pinecone":
            logger.warning(
                "OpenRouter embeddings unavailable - switching to Pinecone hosted (%s).",
                PINECONE_EMBEDDING_MODEL,
            )
            self.provider = "pinecone"

    def switch_to_local(self) -> None:
        if self.provider_tag != "local":
            logger.warning(
                "Fallback embeddings unavailable - switching to local (%s).",
                self.local_model_name,
            )
            self.provider = "local"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents in batches. input_type='passage'."""
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_with_fallback(batch, input_type="passage"))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text. input_type='query'."""
        return self._embed_with_fallback([text], input_type="query")[0]

    def validate_dimensions(self, vectors: List[List[float]]) -> None:
        expected = self.dimension
        for vec in vectors:
            if len(vec) != expected:
                raise EmbeddingDimensionMismatch(
                    f"Embedding dimension mismatch for provider '{self.provider_tag}': "
                    f"expected {expected}, got {len(vec)}. Check the provider's configured "
                    f"dimension (EMBEDDING_DIMENSIONS in config.py) and the Pinecone index "
                    f"dimension before inserting or querying vectors."
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed_with_fallback(self, texts: List[str], input_type: str) -> List[List[float]]:
        # Try the current provider first.
        try:
            vectors = self._embed_batch(texts, input_type)
            self.validate_dimensions(vectors)
            return _to_float_vectors(vectors)
        except EmbeddingError as exc:
            if self.provider_tag == "local":
                # Local is the last resort — no further fallback.
                raise

        # Fallback chain: openrouter -> pinecone -> local
        if self.provider_tag == "openrouter":
            logger.warning("OpenRouter embedding failed; falling back to Pinecone.")
            self.switch_to_pinecone()
            try:
                vectors = self._embed_batch(texts, input_type)
                self.validate_dimensions(vectors)
                return _to_float_vectors(vectors)
            except EmbeddingError as exc:
                logger.warning("Pinecone fallback also failed (%s); falling back to local.", exc)

        if self.provider_tag in ("openrouter", "pinecone"):
            logger.warning("Falling back to local Sentence-Transformers model.")
            self.switch_to_local()
            vectors = self._embed_batch(texts, input_type)
            self.validate_dimensions(vectors)
            return _to_float_vectors(vectors)

        raise EmbeddingError("Embedding failed on all providers.")

    def _embed_batch(self, texts: List[str], input_type: str) -> List[List[float]]:
        if self.provider_tag == "openrouter":
            return self._embed_openrouter(texts, input_type)
        if self.provider_tag == "pinecone":
            return self._embed_pinecone(texts, input_type)
        return self._embed_local(texts)

    # ------------------------------------------------------------------
    # OpenRouter embeddings (optional)
    # ------------------------------------------------------------------
    def _embed_openrouter(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not OPENROUTER_API_KEY:
            raise EmbeddingUnavailableError("OPENROUTER_API_KEY is not configured.")

        url = f"{OPENROUTER_BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if SITE_URL:
            headers["HTTP-Referer"] = SITE_URL
        if SITE_NAME:
            headers["X-OpenRouter-Title"] = SITE_NAME

        or_input_type = "search_document" if input_type == "passage" else "search_query"

        payload = {
            "model": self.openrouter_model,
            "input": texts,
            "encoding_format": "float",
            "input_type": or_input_type,
            "dimensions": self.dimension,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429:
                    last_error = EmbeddingUnavailableError(
                        f"Rate limited (429) on {self.openrouter_model}."
                    )
                    wait = min(2 ** attempt, 8)
                    logger.warning("OpenRouter rate limited; retrying in %ss (attempt %d/%d).",
                                   wait, attempt, self.MAX_ATTEMPTS)
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    raise EmbeddingError(
                        f"OpenRouter embeddings rejected request "
                        f"(HTTP {response.status_code}): {response.text[:300]}"
                    )
                response.raise_for_status()
                data = response.json()
                items = data.get("data")
                if not items or len(items) != len(texts):
                    raise EmbeddingError("OpenRouter returned an unexpected embeddings response shape.")
                ordered = sorted(items, key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in ordered]
                if any(v is None for v in vectors):
                    raise EmbeddingError("OpenRouter returned a null embedding vector.")
                return _to_float_vectors(vectors)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = EmbeddingUnavailableError(f"OpenRouter embedding network error: {exc}")
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(min(2 ** attempt, 8))
            except requests.exceptions.RequestException as exc:
                raise EmbeddingUnavailableError(f"OpenRouter embedding request failed: {exc}") from exc

        raise last_error or EmbeddingUnavailableError("OpenRouter embedding failed for unknown reasons.")

    # ------------------------------------------------------------------
    # Pinecone hosted embeddings (optional)
    # ------------------------------------------------------------------
    def _get_pinecone_client(self):
        if self._pc is None:
            from pinecone import Pinecone
            if not PINECONE_API_KEY:
                raise EmbeddingUnavailableError("PINECONE_API_KEY is not configured.")
            self._pc = Pinecone(api_key=PINECONE_API_KEY)
        return self._pc

    def _embed_pinecone(self, texts: List[str], input_type: str) -> List[List[float]]:
        pc = self._get_pinecone_client()

        params = {
            "input_type": input_type,  # REQUIRED: 'passage' or 'query'
            "truncate": "END",
            "dimensions": self.dimension,
        }

        last_error = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = pc.inference.embed(
                    model=PINECONE_EMBEDDING_MODEL,
                    inputs=texts,
                    parameters=params,
                )
                vectors = [list(map(float, item.values)) for item in response]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Pinecone returned unexpected number of embeddings.")
                return _to_float_vectors(vectors)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "timeout" in msg:
                    wait = min(2 ** attempt, 8)
                    logger.warning("Pinecone embedding retry in %ss (attempt %d/%d): %s",
                                   wait, attempt, self.MAX_ATTEMPTS, exc)
                    time.sleep(wait)
                    continue
                raise EmbeddingUnavailableError(f"Pinecone embedding failed: {exc}") from exc

        raise EmbeddingUnavailableError(f"Pinecone embedding failed after retries: {last_error}")

    # ------------------------------------------------------------------
    # Local Sentence-Transformers (default)
    # ------------------------------------------------------------------
    def _get_local_model(self):
        if self._local_model is None:
            with self._local_lock:
                if self._local_model is None:
                    try:
                        sentence_transformers = importlib.import_module("sentence_transformers")
                    except ModuleNotFoundError as exc:
                        raise EmbeddingUnavailableError(
                            "The local embedding provider requires sentence-transformers. "
                            "Install it with: pip install sentence-transformers"
                        ) from exc
                    logger.info("Loading local embedding model '%s'...", self.local_model_name)
                    self._local_model = sentence_transformers.SentenceTransformer(self.local_model_name)
        return self._local_model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        try:
            model = self._get_local_model()
            vectors = model.encode(texts, show_progress_bar=False)
            return _to_float_vectors(vectors)
        except Exception as exc:
            raise EmbeddingError(f"Local embedding model failed: {exc}") from exc


# --------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------
_embedding_service: Optional[EmbeddingService] = None
_embedding_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        with _embedding_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService()
    return _embedding_service