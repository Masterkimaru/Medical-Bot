"""On-site medical professional lookup (MongoDB) - NLTK-free.

Replaces the PorterStemmer/stopwords pipeline with the lightweight,
negation-preserving tokenizer in src/text_utils.py. The special_cases keys are
the stemmed tokens produced by that tokenizer (compatible with the old
Porter-stemmed keys such as 'allerg', 'palpitat', 'hypertens').
"""
import logging
import os
from typing import List

from pymongo import MongoClient

from src.config import MONGO_URI, MONGODB_DB_NAME
from src.text_utils import light_stem, tokenize

logger = logging.getLogger(__name__)

_client: MongoClient = None


def _get_collection():
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGODB_URI is not configured.")
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[MONGODB_DB_NAME]["medical_professionals"]


def _ensure_text_index(collection) -> None:
    collection.create_index(
        [("name", "text"), ("specialty", "text"), ("keywords", "text")],
        default_language="english",
        name="text_search_idx",
    )


# Special cases mapping (stemmed tokens -> specialties)
special_cases = {
    # Dermatology/Allergy
    "rash": ["Dermatology", "Allergy"],
    "itch": ["Dermatology", "Allergy"],
    "skin": ["Dermatology"],
    "allerg": ["Allergy"],
    "hive": ["Dermatology", "Allergy"],
    "eczema": ["Dermatology"],
    # Cardiology
    "chest": ["Cardiology", "Emergency Medicine"],
    "heart": ["Cardiology"],
    "palpitat": ["Cardiology"],
    "hypertens": ["Cardiology"],
    "bp": ["Cardiology"],
    # Neurology
    "headach": ["Neurology"],
    "migrain": ["Neurology"],
    "seizur": ["Neurology"],
    "stroke": ["Neurology"],
    "numb": ["Neurology"],
    # Pediatrics
    "child": ["Pediatrics"],
    "infant": ["Pediatrics"],
    "baby": ["Pediatrics"],
    "vaccin": ["Pediatrics"],
    # Orthopedics
    "bone": ["Orthopedic Surgery"],
    "joint": ["Orthopedic Surgery", "Rheumatology"],
    "fractur": ["Orthopedic Surgery"],
    "back": ["Orthopedic Surgery"],
    # OB/GYN
    "pregn": ["Obstetrics & Gynecology"],
    "period": ["Obstetrics & Gynecology"],
    "menstru": ["Obstetrics & Gynecology"],
    "pelvic": ["Obstetrics & Gynecology"],
    # Gastroenterology
    "stomach": ["Gastroenterology"],
    "diarrh": ["Gastroenterology"],
    "constip": ["Gastroenterology"],
    "ibs": ["Gastroenterology"],
    # Respiratory
    "cough": ["Pulmonology", "Family Medicine"],
    "breath": ["Pulmonology", "Emergency Medicine"],
    "asthma": ["Pulmonology", "Allergy & Immunology"],
    "wheez": ["Pulmonology"],
    # Mental Health
    "depress": ["Psychiatry"],
    "anxiet": ["Psychiatry"],
    "stress": ["Psychiatry"],
    "mental": ["Psychiatry"],
    # Endocrinology
    "diabet": ["Endocrinology"],
    "thyroid": ["Endocrinology"],
    "pcos": ["Endocrinology"],
    # Urology
    "urin": ["Urology"],
    "prostat": ["Urology"],
    "bladder": ["Urology"],
    # Ophthalmology
    "eye": ["Ophthalmology"],
    "vision": ["Ophthalmology"],
    "glaucoma": ["Ophthalmology"],
    # ENT
    "ear": ["ENT (Otolaryngology)"],
    "nose": ["ENT (Otolaryngology)"],
    "throat": ["ENT (Otolaryngology)"],
    "sinus": ["ENT (Otolaryngology)"],
    # Emergency
    "emerg": ["Emergency Medicine"],
    "trauma": ["Emergency Medicine"],
    "accid": ["Emergency Medicine"],
    # Infectious Diseases
    "fever": ["Infectious Diseases", "Family Medicine"],
    "infect": ["Infectious Diseases"],
    # Rheumatology
    "arthrit": ["Rheumatology"],
    "lupus": ["Rheumatology"],
    "inflamm": ["Rheumatology"],
    # Hematology
    "anem": ["Hematology"],
    "bleed": ["Hematology"],
    # General/Family Medicine
    "flu": ["Family Medicine"],
    "cold": ["Family Medicine"],
    "check": ["Family Medicine"],
}


def _match_special_case_token(token: str) -> bool:
    """True if the token equals a special-case key or is its stem."""
    return token in special_cases or light_stem(token) in special_cases


def preprocess_text(text: str) -> List[str]:
    """Tokenize and lightly stem while preserving negations (replaces NLTK)."""
    return tokenize(text)


def find_relevant_professionals(query: str, max_results: int = 3) -> list:
    collection = _get_collection()
    _ensure_text_index(collection)

    processed = preprocess_text(query)
    text_search = " ".join(processed)

    # 1) Full-text search
    text_results = []
    try:
        text_cursor = collection.find(
            {"$text": {"$search": text_search, "$language": "en",
                        "$caseSensitive": False, "$diacriticSensitive": False}},
            {"score": {"$meta": "textScore"}},
        ).sort([("score", {"$meta": "textScore"})])
        text_results = list(text_cursor)
    except Exception as exc:
        logger.warning("MongoDB text search failed: %s", exc)

    # 2) Special-cases specialty lookup (raw token OR its light stem)
    specs = set()
    for token in processed:
        if _match_special_case_token(token):
            specs.update(special_cases[light_stem(token)])

    spec_results = []
    if specs:
        try:
            spec_results = list(collection.find({"specialty": {"$in": list(specs)}}))
        except Exception as exc:
            logger.warning("MongoDB specialty lookup failed: %s", exc)

    # 3) Merge & dedupe
    combined = text_results + spec_results
    seen = set()
    final = []
    for doc in combined:
        if doc["_id"] not in seen:
            final.append(doc)
            seen.add(doc["_id"])
        if len(final) >= max_results:
            break

    # Fallback: match on specialty names
    if not final:
        try:
            all_specs = collection.distinct("specialty")
            query_specs = [spec for spec in all_specs
                           if any(stem in spec.lower() for stem in (light_stem(t) for t in processed))]
            if query_specs:
                fallback = collection.find({"specialty": {"$in": query_specs}})
                for doc in fallback:
                    if doc["_id"] not in seen:
                        final.append(doc)
                        seen.add(doc["_id"])
                    if len(final) >= max_results:
                        break
        except Exception as exc:
            logger.warning("MongoDB specialty fallback failed: %s", exc)

    # Final fallback: family/general practice
    if not final:
        try:
            fallback = collection.find(
                {"specialty": {"$in": ["Family Medicine", "General Practice"]}}
            ).limit(max_results)
            for doc in fallback:
                if doc["_id"] not in seen:
                    final.append(doc)
                    seen.add(doc["_id"])
                if len(final) >= max_results:
                    break
        except Exception as exc:
            logger.warning("MongoDB general-practice fallback failed: %s", exc)

    return final
