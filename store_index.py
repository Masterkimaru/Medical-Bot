"""Indexing script: load PDFs, embed via the EmbeddingService, upsert to Pinecone.

Safe-index rules:
- Uses the index for the CONFIGURED embedding provider (EMBEDDING_PROVIDER env var).
- Creates the index only if missing; verifies dimension compatibility before upsert.
- NEVER deletes or recreates an existing index.
- Batch embedding + batch upsert, with post-upsert verification.

Run once per knowledge-base change:
    EMBEDDING_PROVIDER=openrouter python store_index.py
"""
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from src.config import (
    EMBEDDING_PROVIDER,
    PINECONE_INDEX_LOCAL,
    PINECONE_INDEX_OPENROUTER,
)
from src.embedding_service import get_embedding_service
from src.helper import load_pdf_file, text_split
from src.vector_service import VectorService, VectorServiceError, get_vector_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("store_index")


def _probe_embedding_dimension(service) -> None:
    """Fail fast if the configured dimension does not match the real model output."""
    try:
        probe = service.embed_query("dimension probe")
    except Exception as exc:
        raise SystemExit(
            f"\n❌ Embedding probe failed: {exc}\n"
            f"   Check that the model '{service.model_name}' is reachable with the "
            f"current provider/API key (free-tier models are frequently rate-limited)."
        )
    actual = len(probe)
    expected = service.dimension
    print(f"   Embedding model outputs {actual} dims; config expects {expected}.")
    if actual != expected:
        raise SystemExit(
            f"\n❌ Dimension mismatch between model and config: "
            f"model produced {actual}, config says {expected}.\n"
            f"   Fix OPENROUTER_EMBEDDING_DIM (or LOCAL model dim) in your .env, then create "
            f"a NEW index with the correct dimension. Existing indexes are never modified."
        )


def main():
    load_dotenv()

    service = get_embedding_service()
    vector_service: VectorService = get_vector_service()

    index_name = (
        PINECONE_INDEX_OPENROUTER if EMBEDDING_PROVIDER == "openrouter" else PINECONE_INDEX_LOCAL
    )

    data_folder = Path(__file__).parent / "Data"
    if not data_folder.exists():
        logger.error("Data folder not found at %s - nothing to index.", data_folder)
        return

    print(f"\n1. Loading PDF documents from {data_folder} ...")
    extracted_data = load_pdf_file(data=str(data_folder))
    print(f"   Loaded {len(extracted_data)} document pages.")
    if not extracted_data:
        print("   No pages extracted - check that Data/ contains valid PDFs.")
        return

    print("\n2. Splitting text into chunks...")
    text_chunks = text_split(extracted_data)
    print(f"   Produced {len(text_chunks)} chunks.")
    if not text_chunks:
        print("   No chunks to index - exiting.")
        return

    # Diagnostic: prove that chunks actually contain text before embedding.
    sample = text_chunks[0]
    sample_text = (sample.page_content or "").strip()
    print(f"   Sample chunk preview: {sample_text[:200]!r}")
    print(f"   Sample metadata keys: {list((sample.metadata or {}).keys())}")
    non_empty = sum(1 for c in text_chunks if (c.page_content or "").strip())
    print(f"   Non-empty chunks: {non_empty}/{len(text_chunks)}")
    if non_empty == 0:
        print("   ❌ All chunks are empty — fix load_pdf_file / text_split before indexing.")
        return

    print(f"\n3. Embedding provider: '{service.provider_tag}' "
          f"(model={service.model_name}, dim={service.dimension})")
    _probe_embedding_dimension(service)

    print(f"\n4. Ensuring Pinecone index '{index_name}'...")
    try:
        vector_service.ensure_index(index_name)
    except VectorServiceError as exc:
        print(f"\n❌ {exc}")
        print("\nMIGRATION GUIDE: never delete or overwrite an existing index. If the existing "
              "index was built with a different embedding model, create a NEW index with the "
              "correct dimension for this model and re-run this script pointing at it "
              "(see PINECONE_INDEX_OPENROUTER / PINECONE_INDEX_LOCAL in .env.example).")
        raise SystemExit(1)

    print("\n5. Upserting documents to Pinecone (batched + verified)...")
    start = time.time()
    try:
        count = vector_service.upsert_documents(text_chunks)
    except VectorServiceError as exc:
        print(f"\n❌ {exc}")
        raise SystemExit(1)

    print(f"\n6. Indexing completed successfully: {count} chunks in {time.time() - start:.1f}s.")

    stats = vector_service.index_stats(index_name)
    total = stats.get("total_vector_count", 0)
    print(f"   Total vectors in index '{index_name}': {total}")
    print(f"   Namespaces: {stats.get('namespaces', {})}")
    if total == 0:
        print("\n⚠️  Index is still empty after a verified upsert — this should not happen. "
              "Re-run with logging at DEBUG for details.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()