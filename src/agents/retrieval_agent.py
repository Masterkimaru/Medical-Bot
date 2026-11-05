"""Retrieval agent: queries Pinecone through the VectorService."""
import logging
from typing import List, Optional

from src.agents.base import BaseAgent
from src.vector_service import VectorServiceError, get_vector_service

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """Retrieves relevant knowledge-base context from Pinecone.

    Note: this agent performs retrieval only; it never calls an LLM, so it
    cannot fail because of a provider outage (embedding fallback is handled
    inside the embedding service / vector service).
    """

    name = "retrieval"
    role = "medical"  # unused: no LLM calls

    def retrieve(self, query: str, k: int = 2) -> List:
        """Return the top-k documents for a query (empty list on any failure)."""
        try:
            store = get_vector_service().get_store()
            docs = store.similarity_search(query, k=k)
            logger.info("Retrieved %d documents for query (provider=%s).",
                        len(docs), get_vector_service().service.provider_tag)
            return docs
        except VectorServiceError as exc:
            # Configuration problem (e.g. dimension mismatch) - surfaced, not swallowed.
            logger.error("Vector service configuration error: %s", exc)
            raise
        except Exception as exc:
            logger.warning("Pinecone retrieval failed (degraded to no-context): %s", exc)
            return []
