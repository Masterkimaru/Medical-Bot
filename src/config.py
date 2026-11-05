"""Central configuration for MediBot.

All configuration is read from environment variables (see .env.example).
No credentials are ever hardcoded here.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get(name: str, default=None, required: bool = False):
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# --------------------------------------------------------------------------
# Pinecone
# --------------------------------------------------------------------------
PINECONE_API_KEY = _get("PINECONE_API_KEY", "")

# Separate indexes per embedding model so vectors are NEVER cross-queried
# with embeddings from an incompatible model.
PINECONE_INDEX_OPENROUTER = _get("PINECONE_INDEX_OPENROUTER", "doctorbot-nemotron-2048")
PINECONE_INDEX_LOCAL = _get("PINECONE_INDEX_LOCAL", "doctorbot-minilm-384")

PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "openrouter").lower()
OPENROUTER_EMBEDDING_MODEL = _get("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
OPENROUTER_EMBEDDING_DIM = int(_get("OPENROUTER_EMBEDDING_DIM", "2048"))
LOCAL_EMBEDDING_MODEL = _get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(_get("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_TIMEOUT = float(_get("EMBEDDING_TIMEOUT", "30"))

# Expected dimensions per provider. Validated against the Pinecone index
# before any insert or query.
EMBEDDING_DIMENSIONS = {
    "openrouter": OPENROUTER_EMBEDDING_DIM,
    "local": 384,  # all-MiniLM-L6-v2
}

# --------------------------------------------------------------------------
# LLM providers (OpenRouter primary, Groq fallback)
# --------------------------------------------------------------------------
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = _get("SITE_URL", "")
SITE_NAME = _get("SITE_NAME", "MediBot")

GROQ_API_KEY = _get("GROQ_API_KEY", "")

MEDICAL_LLM_MODEL = _get("MEDICAL_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
MEDICAL_FALLBACK_LLM_MODEL = _get("MEDICAL_FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")
MENTAL_HEALTH_LLM_MODEL = _get("MENTAL_HEALTH_LLM_MODEL", "ainclusionai/ling-3.0-flash:free")
MENTAL_HEALTH_FALLBACK_LLM_MODEL = _get("MENTAL_HEALTH_FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")
VISION_LLM_MODEL = _get("VISION_LLM_MODEL", "nvidia/nemotron-3.5-content-safety:free")

LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(_get("LLM_MAX_RETRIES", "2"))

# Provider chain per agent role: index 0 is primary, the rest are failovers.
ROLE_LLM_CHAINS = {
    "medical": ["openrouter", "groq"],
    "general_health": ["openrouter", "groq"],
    "mental_health": ["openrouter", "groq"],
    "validation": ["openrouter"],
    # Vision requires a multimodal model; OpenRouter only.
    "vision": ["openrouter"],
}

# --------------------------------------------------------------------------
# MongoDB (on-site professionals)
# --------------------------------------------------------------------------
MONGO_URI = _get("MONGODB_URI", "")
MONGODB_DB_NAME = _get("MONGODB_DB_NAME", "medi_bot")

# --------------------------------------------------------------------------
# Behaviour toggles
# --------------------------------------------------------------------------
VALIDATION_ENABLED = _get("VALIDATION_ENABLED", "false").lower() == "true"
PORT = int(_get("PORT", "5000"))
