"""On-site medical professional lookup (JSON-based) - NLTK-free.
Reads directly from meds.json instead of MongoDB.
"""
import json
import os
import logging
from typing import List, Dict, Any
from src.text_utils import light_stem, tokenize

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Load meds.json once at startup and cache in memory
# --------------------------------------------------------------------------
_MEDS_DATA: List[Dict[str, Any]] = []
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming meds.json is in the root directory (one level up from src/)
_MEDS_FILE_PATH = os.path.join(_BASE_DIR, '..', 'meds.json')
if not os.path.exists(_MEDS_FILE_PATH):
    # Fallback to current working directory if not found relative to script
    _MEDS_FILE_PATH = os.path.join(os.getcwd(), 'meds.json')

def _load_meds() -> List[Dict[str, Any]]:
    global _MEDS_DATA
    if not _MEDS_DATA:
        try:
            with open(_MEDS_FILE_PATH, 'r', encoding='utf-8') as f:
                _MEDS_DATA = json.load(f)
            logger.info(f"Successfully loaded {len(_MEDS_DATA)} medical professionals from meds.json")
        except Exception as e:
            logger.error(f"Failed to load meds.json from {_MEDS_FILE_PATH}: {e}")
            _MEDS_DATA = []
    return _MEDS_DATA

# --------------------------------------------------------------------------
# 2. Special cases mapping (FIXED: All trailing spaces removed)
# --------------------------------------------------------------------------
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
    """Tokenize and lightly stem while preserving negations."""
    return tokenize(text)

# --------------------------------------------------------------------------
# 3. Core Lookup Logic (Replaces MongoDB queries)
# --------------------------------------------------------------------------
def find_relevant_professionals(query: str, max_results: int = 3) -> list:
    docs = _load_meds()
    if not docs:
        return []
        
    processed = preprocess_text(query)
    processed_stems = [light_stem(t) for t in processed]
    
    # 1) Keyword & Specialty matching with a scoring system
    scored_results = []
    for doc in docs:
        score = 0
        doc_keywords = [k.lower() for k in doc.get("keywords", [])]
        doc_specialty = doc.get("specialty", "").lower()
        
        for stem in processed_stems:
            # Boost score if the stem matches the specialty directly
            if stem in doc_specialty:
                score += 2
            # Check against the doctor's specific keywords
            for kw in doc_keywords:
                if stem in kw or kw in stem:
                    score += 1
                    break  # Count once per keyword match per stem
        
        if score > 0:
            scored_results.append((score, doc))
            
    # Sort by score descending and take top results
    scored_results.sort(key=lambda x: x[0], reverse=True)
    final = [doc for _, doc in scored_results[:max_results]]
    
    # 2) Special-cases specialty lookup (if we haven't found enough doctors yet)
    if len(final) < max_results:
        specs = set()
        for token in processed:
            stem = light_stem(token)
            if stem in special_cases:
                specs.update(special_cases[stem])
            elif token in special_cases:
                specs.update(special_cases[token])
                
        if specs:
            for doc in docs:
                if doc.get("specialty") in specs:
                    # Avoid duplicates
                    if not any(d.get("name") == doc.get("name") for d in final):
                        final.append(doc)
                        if len(final) >= max_results:
                            break
                            
    # 3) Fallback: Family/General practice if nothing matched
    if not final:
        for doc in docs:
            if doc.get("specialty") in ["Family Medicine", "General Practice"]:
                if not any(d.get("name") == doc.get("name") for d in final):
                    final.append(doc)
                    if len(final) >= max_results:
                        break
                        
    return final