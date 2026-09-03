from pymongo import MongoClient
import os
from functools import lru_cache
import gc

# ===== LAZY LOAD NLTK =====
_nltk_initialized = False
_ps = None
_stop_words = None

def _init_nltk():
    """Initialize NLTK only when needed"""
    global _nltk_initialized, _ps, _stop_words
    if _nltk_initialized:
        return
    
    try:
        import nltk
        # Ensure NLTK data is available
        try:
            from nltk.data import find
            find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        from nltk.stem import PorterStemmer
        from nltk.corpus import stopwords
        
        _ps = PorterStemmer()
        _stop_words = set(stopwords.words('english'))
        _nltk_initialized = True
        print("✅ NLTK initialized in helper_pros")
    except Exception as e:
        print(f"⚠️ NLTK initialization warning: {e}")
        _nltk_initialized = True  # Prevent retry loops

# ===== LAZY LOAD MONGODB =====
_mongo_client = None
_pros_col = None

def get_pros_collection():
    """Lazy load MongoDB connection"""
    global _mongo_client, _pros_col
    if _pros_col is None:
        try:
            MONGO_URI = os.environ.get("MONGODB_URI")
            if not MONGO_URI:
                print("⚠️ MONGODB_URI not set")
                return None
            
            DB_NAME = os.environ.get("MONGODB_DB_NAME", "medi_bot")
            _mongo_client = MongoClient(
                MONGO_URI, 
                maxPoolSize=1,  # Limit connections
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            _pros_col = _mongo_client[DB_NAME]["medical_professionals"]
            
            # Create index only if it doesn't exist (check first)
            try:
                existing_indexes = _pros_col.index_information()
                if 'text_search_idx' not in existing_indexes:
                    _pros_col.create_index(
                        [("name", "text"), ("specialty", "text"), ("keywords", "text")],
                        default_language="english",
                        name="text_search_idx"
                    )
            except Exception as e:
                print(f"⚠️ Index creation warning: {e}")
            
            print("✅ MongoDB connected")
        except Exception as e:
            print(f"⚠️ MongoDB connection error: {e}")
            return None
    
    return _pros_col

@lru_cache(maxsize=128)
def preprocess_text_cached(text):
    """Cached version of text preprocessing"""
    _init_nltk()
    if not _ps or not _stop_words:
        return []
    
    try:
        import nltk
        tokens = nltk.word_tokenize(text.lower())
        filtered = [_ps.stem(w) for w in tokens if w.isalnum() and w not in _stop_words]
        return filtered
    except Exception as e:
        print(f"⚠️ Text preprocessing error: {e}")
        return []

def preprocess_text(text):
    """Tokenize, stem, and remove stopwords using NLTK"""
    return preprocess_text_cached(text)

# Special cases mapping (keep as is)
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

def find_relevant_professionals(query, max_results=3):
    """Find relevant professionals with optimized memory usage"""
    try:
        # Initialize NLTK if needed
        _init_nltk()
        
        # Get collection
        pros_col = get_pros_collection()
        if pros_col is None:
            return []
        
        # Preprocess query
        processed = preprocess_text(query)
        if not processed:
            return []
        
        text_search = " ".join(processed[:10])  # Limit search terms

        # Limit search results to reduce memory
        text_cursor = pros_col.find(
            {"$text": {"$search": text_search, "$language": "en", "$caseSensitive": False, "$diacriticSensitive": False}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(max_results * 2)
        
        text_results = list(text_cursor)

        # Special-cases specialty lookup
        specs = set()
        for token in processed[:10]:  # Limit tokens
            if token in special_cases:
                specs.update(special_cases[token])

        spec_results = []
        if specs:
            spec_cursor = pros_col.find({"specialty": {"$in": list(specs)}}).limit(max_results * 2)
            spec_results = list(spec_cursor)

        # Merge & dedupe
        combined = text_results + spec_results
        seen = set()
        final = []
        for doc in combined:
            if doc['_id'] not in seen:
                # Only keep essential fields to reduce memory
                final.append({
                    'name': doc.get('name', 'Unknown'),
                    'specialty': doc.get('specialty', 'General'),
                    'phone': doc.get('phone', 'N/A'),
                    '_id': doc['_id']
                })
                seen.add(doc['_id'])
            if len(final) >= max_results:
                break

        # Fallback: match on stemmed specialty names
        if not final:
            all_specs = pros_col.distinct("specialty")
            query_specs = []
            for spec in all_specs[:20]:  # Limit to 20 specialties
                if any(ps.stem(w) in spec.lower() for w in processed[:5]):  # Limit tokens
                    query_specs.append(spec)
            
            if query_specs:
                fallback_cursor = pros_col.find({"specialty": {"$in": query_specs}}).limit(max_results)
                for doc in fallback_cursor:
                    if doc['_id'] not in seen:
                        final.append({
                            'name': doc.get('name', 'Unknown'),
                            'specialty': doc.get('specialty', 'General'),
                            'phone': doc.get('phone', 'N/A'),
                            '_id': doc['_id']
                        })
                        seen.add(doc['_id'])
                    if len(final) >= max_results:
                        break

        # Final fallback: family/general practice
        if not final:
            fallback_cursor = pros_col.find({"specialty": {"$in": ["Family Medicine", "General Practice"]}}).limit(max_results)
            for doc in fallback_cursor:
                if doc['_id'] not in seen:
                    final.append({
                        'name': doc.get('name', 'Unknown'),
                        'specialty': doc.get('specialty', 'General'),
                        'phone': doc.get('phone', 'N/A'),
                        '_id': doc['_id']
                    })
                    seen.add(doc['_id'])
                if len(final) >= max_results:
                    break

        # Clean up
        gc.collect()
        
        return final
        
    except Exception as e:
        print(f"⚠️ Error finding professionals: {e}")
        gc.collect()
        return []

# Clear cache when module is unloaded
import atexit
@atexit.register
def cleanup():
    preprocess_text_cached.cache_clear()
    gc.collect()
