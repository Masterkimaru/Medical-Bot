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
EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "local").lower()
OPENROUTER_EMBEDDING_MODEL = _get("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
OPENROUTER_EMBEDDING_DIM = int(_get("OPENROUTER_EMBEDDING_DIM", "2048"))
LOCAL_EMBEDDING_MODEL = _get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(_get("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_TIMEOUT = float(_get("EMBEDDING_TIMEOUT", "30"))

# Expected dimensions per provider. Validated against the Pinecone index
# before any insert or query.
EMBEDDING_DIMENSIONS = {
    "pinecone": 1024,
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

# --- Model selection -------------------------------------------------------
#
# Use NON-reasoning models for chat roles. Reasoning models (nvidia/nemotron,
# deepseek-r1, o1-style, etc.) spend the completion-token budget on an internal
# chain-of-thought phase *before* emitting any answer. With a small max_tokens
# they hit finish_reason="length" while content is still None, which the
# caller sees as an empty completion. That was the original bug.
#
# google/gemma-4-26b-a4b-it:free is a non-reasoning free-tier OpenRouter model
# and is safe with the max_tokens below. If you prefer higher quality, swap
# the primary to "openai/gpt-4o-mini" or "meta-llama/llama-3.3-70b-instruct"
# (both cheap, non-reasoning, reliable for medical phrasing).
#
# The Groq fallback MUST be a valid Groq model id (see
# https://console.groq.com/docs/models). Common choices:
#   llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it,
#   mixtral-8x7b-32768.
# Do NOT put an OpenRouter ":free" id here — Groq will 404.
# --------------------------------------------------------------------------
MEDICAL_LLM_MODEL = _get("MEDICAL_LLM_MODEL", "google/gemma-4-26b-a4b-it:free")
MEDICAL_FALLBACK_LLM_MODEL = _get("MEDICAL_FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")

MENTAL_HEALTH_LLM_MODEL = _get("MENTAL_HEALTH_LLM_MODEL", "google/gemma-4-26b-a4b-it:free")
MENTAL_HEALTH_FALLBACK_LLM_MODEL = _get("MENTAL_HEALTH_FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")

# Vision needs a multimodal model. "google/gemini-2.0-flash-exp:free" is a real
# vision-capable free model on OpenRouter. (nemotron-*-content-safety is a
# classifier, not a VLM.)
VISION_LLM_MODEL = _get("VISION_LLM_MODEL", "google/gemini-2.0-flash-exp:free")

# 1000 is too small: reasoning models need headroom, and long answers get
# truncated on any model. 4000 is a safe default for both chat and fallback.
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "4000"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(_get("LLM_MAX_RETRIES", "1"))

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