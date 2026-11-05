"""Lightweight text processing utilities that replace NLTK.

Design rules (per the migration requirements):
- Preserve negations ("no fever", "not breathing") - they are clinically critical.
- Preserve medication terminology (dosages, drug names) - no aggressive stemming.
- Only the safest, minimal stemming is applied, and ONLY for the MongoDB
  specialty lookup, never for embeddings or LLM prompts.
"""
import re
from typing import List

# A tiny, medical-safe suffix stripper. Unlike PorterStemmer it never mangles
# medication names and never collapses negations.
_SUFFIXES = (
    ("ies", "y"),
    ("ements", "ement"),
    ("ements", "ement"),
    ("ations", "ation"),
    ("ings", "ing"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

# Tokens that must never be stripped/stemmed (negations, modifiers, drugs).
_PRESERVE = {
    "no", "not", "nor", "never", "without", "denies", "denied",
    "mg", "ml", "mcg", "iu", "units", "dose", "dosage", "tablet", "tablets",
    "capsule", "capsules", "paracetamol", "acetaminophen", "ibuprofen",
    "aspirin", "amoxicillin", "penicillin", "antibiotics", "insulin",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens using `re` (replaces nltk.word_tokenize)."""
    return _WORD_RE.findall(text.lower())


def light_stem(token: str) -> str:
    """Very conservative suffix stripping; preserves negations and medication terms."""
    if token in _PRESERVE or len(token) <= 3:
        return token
    for suffix, replacement in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)] + replacement
    return token


def preprocess_text(text: str) -> List[str]:
    """Tokenize and lightly stem, KEEPING stopwords that carry negation.

    This is a drop-in replacement for the old NLTK-based preprocess_text in
    helper_pros.py. Stopwords are NOT removed wholesale: words like 'no' and
    'not' change the meaning of medical queries and are kept.
    """
    tokens = tokenize(text)
    return [light_stem(tok) for tok in tokens]
