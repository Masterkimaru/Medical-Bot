from pymongo import MongoClient
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# MongoDB connection
MONGO_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB_NAME", "medi_bot")
client = MongoClient(MONGO_URI)
pros_col = client[DB_NAME]["medical_professionals"]

# Ensure text index exists for full-text search
pros_col.create_index(
    [
        ("name", "text"),
        ("specialty", "text"),
        ("keywords", "text")
    ],
    default_language="english",
    name="text_search_idx"
)

# Special cases mapping (stemmed tokens)
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
    "check": ["Family Medicine"]
}

def preprocess_text(text):
    """
    Tokenize, stem, and remove stopwords using NLTK.
    """
    tokens = nltk.word_tokenize(text.lower())
    filtered = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words]
    return filtered


def find_relevant_professionals(query, max_results=3):
    processed = preprocess_text(query)
    text_search = " ".join(processed)

    # 1) Full-text search
    text_cursor = pros_col.find(
        {"$text": {"$search": text_search, "$language": "en", "$caseSensitive": False, "$diacriticSensitive": False}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})])
    text_results = list(text_cursor)

    # 2) Special-cases specialty lookup
    specs = set()
    for token in processed:
        if token in special_cases:
            specs.update(special_cases[token])

    spec_results = []
    if specs:
        spec_results = list(pros_col.find({"specialty": {"$in": list(specs)}}))

    # 3) Merge & dedupe
    combined = text_results + spec_results
    seen = set()
    final = []
    for doc in combined:
        if doc['_id'] not in seen:
            final.append(doc)
            seen.add(doc['_id'])
        if len(final) >= max_results:
            break

    # Fallback: match on stemmed specialty names
    if not final:
        all_specs = pros_col.distinct("specialty")
        query_specs = [spec for spec in all_specs if any(ps.stem(w) in spec.lower() for w in processed)]
        if query_specs:
            fallback = pros_col.find({"specialty": {"$in": query_specs}})
            for doc in fallback:
                if doc['_id'] not in seen:
                    final.append(doc)
                    seen.add(doc['_id'])
                if len(final) >= max_results:
                    break

    # Final fallback: family/general practice
    if not final:
        fallback = pros_col.find({"specialty": {"$in": ["Family Medicine", "General Practice"]}}).limit(max_results)
        for doc in fallback:
            if doc['_id'] not in seen:
                final.append(doc)
                seen.add(doc['_id'])
            if len(final) >= max_results:
                break

    return final
