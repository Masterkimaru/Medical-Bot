"""Embedding service: OpenRouter API with a safe local Sentence-Transformers fallback.

Key safety rule: embeddings from different models are NOT interchangeable.
This service tags every embedding request with the provider that generated it,
and the vector service uses that tag to select the matching Pinecone index.
"""
import logging
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
        self.provider = (provider or EMBEDDING_PROVIDER).lower()
        self.openrouter_model = openrouter_model or OPENROUTER_EMBEDDING_MODEL
        self.local_model_name = local_model or LOCAL_EMBEDDING_MODEL
        self.batch_size = max(1, batch_size)
        self.timeout = timeout

        self._local_model = None
        self._local_lock = threading.Lock()
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Provider bookkeeping
    # ------------------------------------------------------------------
    @property
    def provider_tag(self) -> str:
        """Stable identifier used to select the matching Pinecone index."""
        return "openrouter" if self.provider == "openrouter" else "local"

    @property
    def model_name(self) -> str:
        return self.openrouter_model if self.provider_tag == "openrouter" else self.local_model_name

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSIONS[self.provider_tag]

    def switch_to_local(self) -> None:
        """Switch this service to the local fallback provider."""
        if self.provider_tag != "local":
            logger.warning("OpenRouter embeddings unavailable - switching to local fallback (%s).",
                           self.local_model_name)
            self.provider = "local"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents in batches. Runs on the fallback if needed."""
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_with_fallback(batch))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self._embed_with_fallback([text])[0]

    def validate_dimensions(self, vectors: List[List[float]]) -> None:
        """Ensure returned vectors match the configured dimension for this provider."""
        expected = self.dimension
        for vec in vectors:
            if len(vec) != expected:
                raise EmbeddingDimensionMismatch(
                    f"Embedding dimension mismatch for provider '{self.provider_tag}': "
                    f"expected {expected}, got {len(vec)}. Check OPENROUTER_EMBEDDING_DIM and "
                    f"the Pinecone index dimension before inserting or querying vectors."
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed_with_fallback(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self._embed_batch(texts)
            self.validate_dimensions(vectors)
            return vectors
        except EmbeddingError as exc:
            if self.provider_tag == "local":
                raise
            logger.warning("OpenRouter embedding failed (%s); falling back to local model.", exc)
            self.switch_to_local()
            vectors = self._embed_batch(texts)
            self.validate_dimensions(vectors)
            return vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider_tag == "openrouter":
            return self._embed_openrouter(texts)
        return self._embed_local(texts)

    def _embed_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Embed via the OpenRouter /embeddings endpoint (supports batch input)."""
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

        payload = {
            "model": self.openrouter_model,
            "input": texts,
            "encoding_format": "float",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429:
                    last_error = EmbeddingUnavailableError(f"Rate limited (429) on {self.openrouter_model}.")
                    wait = min(2 ** attempt, 8)
                    logger.warning("Embedding rate limited; retrying in %ss (attempt %d/%d).",
                                   wait, attempt, self.MAX_ATTEMPTS)
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    # 4xx other than 429: invalid request / auth / model not available.
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
                return vectors
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = EmbeddingUnavailableError(f"OpenRouter embedding network error: {exc}")
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(min(2 ** attempt, 8))
            except requests.exceptions.RequestException as exc:
                raise EmbeddingUnavailableError(f"OpenRouter embedding request failed: {exc}") from exc

        raise last_error or EmbeddingUnavailableError("OpenRouter embedding failed for unknown reasons.")

    def _get_local_model(self):
        """Lazy-load the local Sentence-Transformers model (once per process)."""
        if self._local_model is None:
            with self._local_lock:
                if self._local_model is None:
                    from sentence_transformers import SentenceTransformer  # heavy import, lazy
                    logger.info("Loading local embedding model '%s'...", self.local_model_name)
                    self._local_model = SentenceTransformer(self.local_model_name)
        return self._local_model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        try:
            model = self._get_local_model()
            vectors = model.encode(texts, show_progress_bar=False)
            return [list(map(float, vec)) for vec in vectors]
        except Exception as exc:  # model download/load failure
            raise EmbeddingError(f"Local embedding model failed: {exc}") from exc


# --------------------------------------------------------------------------
# Process-wide singleton (clients and models are reused across requests)
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
