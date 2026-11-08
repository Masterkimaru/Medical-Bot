"""Vector service: Pinecone access with per-embedding-model index isolation.

Safety rules:
- One Pinecone index per embedding provider/model. The index dimension is
  verified before every use; vectors are never written to or queried with an
  incompatible model's embeddings.
- Existing indexes are NEVER deleted or recreated. If a dimension mismatch is
  detected, the service raises with migration instructions instead.
- Upserts are verified: if Pinecone accepts the call but the index count does
  not increase, a VectorServiceError is raised instead of silently returning.
- All vector values are coerced to Python float before reaching Pinecone,
  because the Pinecone client rejects int values with PineconeApiTypeError.
"""
import logging
import threading
import time
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_LOCAL,
    PINECONE_INDEX_OPENROUTER,
    PINECONE_REGION,
)
from src.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

# Pinecone-safe metadata value types.
_ALLOWED_META_TYPES = (str, int, float, bool)


class VectorServiceError(Exception):
    """Vector store configuration problem that requires operator action."""


class ServiceEmbeddings(Embeddings):
    """Adapter exposing EmbeddingService through the LangChain Embeddings interface.

    Coerces every vector element to Python float so Pinecone never sees ints.
    """

    def __init__(self, service: EmbeddingService):
        self._service = service

    @property
    def provider_tag(self) -> str:
        return self._service.provider_tag

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._service.embed_documents(texts)
        return [[float(x) for x in vec] for vec in vectors]

    def embed_query(self, text: str) -> List[float]:
        vec = self._service.embed_query(text)
        return [float(x) for x in vec]


def _sanitize_metadata(metadata: Optional[dict]) -> dict:
    """Return a Pinecone-safe metadata dict.

    Drops None values, nested dicts/lists, and keys beginning with '$'
    (all rejected by Pinecone and often silently swallowed by wrappers).
    """
    if not metadata:
        return {}
    clean = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if str(k).startswith("$"):
            continue
        if not isinstance(v, _ALLOWED_META_TYPES):
            # Coerce lists/tuples/dicts into their string form so the record
            # is not rejected. Adjust here if you want different behaviour.
            v = str(v)
        clean[str(k)] = v
    return clean


class VectorService:
    """Manages Pinecone connections and the vector store for the active embedding provider."""

    def __init__(self, service: Optional[EmbeddingService] = None):
        self.service = service or get_embedding_service()
        if not PINECONE_API_KEY:
            raise VectorServiceError("PINECONE_API_KEY is not configured.")
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._store: Optional[PineconeVectorStore] = None
        self._store_provider_tag: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def index_name_for_provider(self, provider_tag: Optional[str] = None) -> str:
        tag = provider_tag or self.service.provider_tag
        return PINECONE_INDEX_OPENROUTER if tag == "openrouter" else PINECONE_INDEX_LOCAL

    def ensure_index(self, index_name: Optional[str] = None) -> str:
        """Ensure the index for the active embedding provider exists and matches dimensions.

        Never deletes or recreates an existing index.
        """
        name = index_name or self.index_name_for_provider()
        expected_dim = self.service.dimension
        existing = self._pc.list_indexes()
        existing_names = existing.names() if hasattr(existing, "names") else existing

        if name not in existing_names:
            logger.info("Creating Pinecone index '%s' (dim=%d, %s/%s)...",
                        name, expected_dim, PINECONE_CLOUD, PINECONE_REGION)
            self._pc.create_index(
                name=name,
                dimension=expected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            # Wait until the index is ready.
            while not self._pc.describe_index(name).status["ready"]:
                time.sleep(2)
            logger.info("Index '%s' is ready.", name)
            return name

        # Index exists: verify its dimension before touching vectors.
        description = self._pc.describe_index(name)
        actual_dim = description.dimension
        if actual_dim != expected_dim:
            raise VectorServiceError(
                f"Dimension mismatch: index '{name}' has dimension {actual_dim}, but embedding model "
                f"'{self.service.model_name}' produces {expected_dim}. Do NOT write or query this index "
                f"with incompatible embeddings. Options: (1) point PINECONE_INDEX_{name.upper()} at a new "
                f"index, (2) create a dedicated index for this model and re-index documents into it "
                f"(python store_index.py with the desired provider), or (3) keep using the matching provider."
            )
        return name

    # ------------------------------------------------------------------
    # Vector store access
    # ------------------------------------------------------------------
    def get_store(self) -> PineconeVectorStore:
        """Get the LangChain vector store bound to the ACTIVE embedding provider.

        Rebuilds the binding automatically if the embedding service has fallen
        back to a different provider (different index + dimensions).
        """
        provider_tag = self.service.provider_tag
        if self._store is None or self._store_provider_tag != provider_tag:
            with self._lock:
                if self._store is None or self._store_provider_tag != provider_tag:
                    index_name = self.ensure_index()
                    index = self._pc.Index(index_name)
                    self._store = PineconeVectorStore(
                        index=index,
                        embedding=ServiceEmbeddings(self.service),
                        text_key="text",
                    )
                    self._store_provider_tag = provider_tag
                    logger.info("Vector store ready: index='%s', provider='%s', model='%s'",
                                index_name, provider_tag, self.service.model_name)
        return self._store

    # ------------------------------------------------------------------
    # Query / upsert
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        return self.get_store().similarity_search(query, k=k)

    def upsert_documents(self, documents: List[Document]) -> int:
        """Embed and upsert documents in batches. Returns number of chunks indexed.

        Validates and verifies the upsert so that a silent no-op cannot occur.
        """
        # 1. Drop anything that would embed to an empty vector.
        valid_docs: List[Document] = []
        dropped_empty = 0
        for d in documents:
            content = (d.page_content or "").strip()
            if not content:
                dropped_empty += 1
                continue
            d.page_content = content
            d.metadata = _sanitize_metadata(d.metadata)
            valid_docs.append(d)

        if dropped_empty:
            logger.warning("Skipped %d empty/whitespace documents before upsert.", dropped_empty)
        if not valid_docs:
            raise VectorServiceError("No non-empty documents to upsert.")

        store = self.get_store()
        index_name = self.index_name_for_provider()

        before = self.index_stats(index_name).get("total_vector_count", 0)
        logger.info("Upserting %d chunks to index '%s' (current count=%d)...",
                    len(valid_docs), index_name, before)

        # 2. add_documents batches embedding (via ServiceEmbeddings) and upserts.
        store.add_documents(valid_docs)

        # 3. Verify the count actually increased (Pinecone is eventually consistent).
        deadline = time.time() + 30.0
        after = before
        while time.time() < deadline:
            time.sleep(2)
            after = self.index_stats(index_name).get("total_vector_count", 0)
            if after > before:
                break

        logger.info("Upsert verified: total_vector_count %d -> %d (attempted=%d).",
                    before, after, len(valid_docs))
        if after <= before:
            raise VectorServiceError(
                f"Upsert reported success but index '{index_name}' count did not increase "
                f"(before={before}, after={after}). Likely causes: embedding dimension mismatch, "
                f"metadata rejection, or embeddings returned empty vectors."
            )
        return len(valid_docs)

    def index_stats(self, index_name: Optional[str] = None) -> dict:
        name = index_name or self.index_name_for_provider()
        return self._pc.Index(name).describe_index_stats()


# --------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------
_vector_service: Optional[VectorService] = None
_vector_lock = threading.Lock()


def get_vector_service() -> VectorService:
    global _vector_service
    if _vector_service is None:
        with _vector_lock:
            if _vector_service is None:
                _vector_service = VectorService()
    return _vector_service