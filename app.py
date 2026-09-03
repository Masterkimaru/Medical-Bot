from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
import re
import requests
import base64
from datetime import datetime
from flask_cors import CORS
import gc
import time
from functools import wraps
from threading import Lock
import logging
import signal
import resource
from functools import lru_cache
import hashlib

# ===== MEMORY OPTIMIZATION =====
# Set memory limits (if on Linux)
try:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))  # 512MB limit
except:
    pass

# More aggressive garbage collection
gc.set_threshold(700, 10, 5)
gc.collect()

# Disable debug logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# ===== NLTK LAZY LOADING =====
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
Path(NLTK_DATA_PATH).mkdir(parents=True, exist_ok=True)

_nltk_loaded = False

def ensure_nltk_data():
    """Lazy load NLTK data only when needed"""
    global _nltk_loaded
    if _nltk_loaded:
        return True
    
    try:
        import nltk
        from nltk.data import find
        try:
            find('tokenizers/punkt')
            _nltk_loaded = True
            return True
        except LookupError:
            pass
            
        # Download only what's needed
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
        # REMOVED wordnet - saves ~100MB memory
        # REMOVED stopwords - not needed if not using in app.py
        _nltk_loaded = True
        print("✅ NLTK data loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ NLTK setup warning: {str(e)}")
        return False

# Call this only when needed
# ensure_nltk_data()  # Don't call here - lazy load

# Import all prompt definitions from src/prompt.py
from src.prompt import (
    system_prompt, 
    emergency_subprompt,
    image_analysis_prompt,
    bmi_interpretation,
    mood_tracking_prompt,
    cbt_exercises_prompt
)

# Import the first aid video helper function
from video_link import get_first_aid_video

# Import helper to find on-site professionals
from src.helper_pros import find_relevant_professionals

from pinecone import Pinecone, data
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.llms.base import LLM
from pydantic import Field

# Initialize Flask app with CORS
app = Flask(__name__)

# Optimize Flask configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

CORS(app, origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://medibot-frontend-eight.vercel.app"
])

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Monkey-patch pinecone.Index to use the new client's index type
import pinecone
pinecone.Index = data.index.Index

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "doctorbot"

# Connect to the Pinecone index
index = pc.Index(index_name)

# Load embeddings and initialize the vector store
embeddings = download_hugging_face_embeddings()

# ===== LAZY LOADING FOR HEAVY INSTANCES =====
_llm_instance = None
_mental_health_llm = None
_vector_store_instance = None
_image_analyzer_instance = None
_llm_lock = Lock()
_mental_health_lock = Lock()
_vector_store_lock = Lock()
_image_analyzer_lock = Lock()

# ===== SIMPLE CACHE FOR LLM RESPONSES =====
_response_cache = {}
_cache_lock = Lock()
MAX_CACHE_SIZE = 15  # Limit cache size

def get_cached_response(query):
    """Get cached response if available"""
    cache_key = hashlib.md5(query[:100].encode()).hexdigest()
    with _cache_lock:
        return _response_cache.get(cache_key)

def cache_response(query, response):
    """Cache a response"""
    cache_key = hashlib.md5(query[:100].encode()).hexdigest()
    with _cache_lock:
        if len(_response_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entry
            _response_cache.pop(next(iter(_response_cache)))
        _response_cache[cache_key] = response

def get_vector_store():
    """Lazy load vector store"""
    global _vector_store_instance
    if _vector_store_instance is None:
        with _vector_store_lock:
            if _vector_store_instance is None:
                _vector_store_instance = LangchainPinecone(
                    index=index,
                    embedding=embeddings,
                    text_key="text"
                )
    return _vector_store_instance

# Define the Deepseek LLM class with reduced token usage
class DeepseekLLM(LLM):
    api_key: str = Field(..., description="OpenRouter API key")
    temperature: float = Field(0.2, description="Sampling temperature")  # Reduced further
    max_tokens: int = Field(500, description="Maximum number of tokens to generate")  # Reduced from 1000
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop=None) -> str:
        # Check cache first
        cached = get_cached_response(prompt)
        if cached:
            return cached
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/nemotron-3.5-lightning:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)  # Reduced timeout
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Cache the response
            cache_response(prompt, content)
            return content
            
        except requests.exceptions.Timeout:
            return "The request timed out. Please try with a shorter question."
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "I'm currently experiencing high load. Please try again in a moment."
    
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

def get_llm():
    """Lazy load LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                _llm_instance = DeepseekLLM(
                    api_key=OPENROUTER_API_KEY, 
                    temperature=0.2, 
                    max_tokens=500
                )
    return _llm_instance

# ===== GOOGLE GEMMA IMAGE ANALYZER (Optimized) =====
class GemmaImageAnalyzer:
    def __init__(self, api_key, site_url="", site_name=""):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if self.site_url:
            self.headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            self.headers["X-OpenRouter-Title"] = self.site_name

    def analyze_image(self, base64_image, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": "nvidia/nemotron-3.5-lightning:free",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "temperature": 0.2,
            "max_tokens": 300  # Reduced from 500
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            if not result or 'choices' not in result:
                return "Unable to analyze image. API returned unexpected format."
            
            if len(result['choices']) == 0:
                return "Unable to analyze image. No response generated."
            
            message = result['choices'][0].get('message', {})
            if not message or 'content' not in message:
                return "Unable to analyze image. No content returned."
            
            return message['content'] or "Image analysis completed but returned empty result."
            
        except requests.exceptions.Timeout:
            return "Image analysis timed out. Please try with a smaller image."
        except requests.exceptions.RequestException as e:
            print(f"❌ Image API Error: {e}")
            return "Image analysis service unavailable. Please try again later."
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            return f"Unable to analyze image: {str(e)}"

def get_image_analyzer():
    """Lazy load image analyzer"""
    global _image_analyzer_instance
    if _image_analyzer_instance is None:
        with _image_analyzer_lock:
            if _image_analyzer_instance is None:
                site_url = os.environ.get("SITE_URL", "")
                site_name = os.environ.get("SITE_NAME", "MediBot")
                _image_analyzer_instance = GemmaImageAnalyzer(
                    api_key=OPENROUTER_API_KEY,
                    site_url=site_url,
                    site_name=site_name
                )
    return _image_analyzer_instance

# Emergency keywords detection (optimized)
EMERGENCY_KEYWORDS = {
    'faint', 'seizure', 'choking', 'bleeding', 'stroke', 'nosebleed',
    'unconscious', 'heart attack', 'overdose', 'cardiac arrest', 'no pulse',
    'not breathing', 'gasping', 'CPR', 'AED', 'chest pain', 'tight chest',
    'unresponsive', 'dizzy', 'confused', 'slurred speech', 'numbness', 'paralyzed',
    'broken bone', 'fracture', 'head injury', 'concussion', 'burn', 'cut', 'sprain',
    'deep wound', 'heavy bleeding', 'open fracture', 'hypothermia', 'heat-stroke',
    'frostbite', 'dehydration', 'drowning', 'allergic reaction', 'anaphylaxis',
    'epipen', 'bee sting', 'snake bite', 'poisoning'
}

def detect_query_category(query):
    """Determine if query is emergency or normal (optimized)"""
    query_lower = query.lower()
    return "emergency" if any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS) else "non-emergency"

# === Qwen LLM for Mental Health (Optimized) ===
class QwenMentalHealthLLM(LLM):
    api_key: str = Field(..., description="OpenRouter API key")
    temperature: float = Field(0.2, description="Sampling temperature")  # Reduced
    max_tokens: int = Field(400, description="Maximum tokens")  # Reduced from 800
    site_url: str = Field("", description="Optional site URL for rankings")
    site_name: str = Field("", description="Optional site title for rankings")

    @property
    def _llm_type(self) -> str:
        return "qwen-next-80b"

    def _call(self, prompt: str, stop=None) -> str:
        # Check cache
        cached = get_cached_response(prompt)
        if cached:
            return cached
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name
        
        payload = {
            "model": "ainclusionai/ling-3.0-flash:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            cache_response(prompt, content)
            return content
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again."
        except Exception as e:
            print(f"❌ Mental Health LLM Error: {e}")
            return "I'm currently experiencing high load. Please try again later."

def get_mental_health_llm():
    """Lazy load mental health LLM"""
    global _mental_health_llm
    if _mental_health_llm is None:
        with _mental_health_lock:
            if _mental_health_llm is None:
                _mental_health_llm = QwenMentalHealthLLM(
                    api_key=OPENROUTER_API_KEY, 
                    temperature=0.2, 
                    max_tokens=400,
                    site_url="https://your-site-url.com",
                    site_name="MediBot"
                )
    return _mental_health_llm

# Clean response function (optimized)
def clean_response(response):
    """Clean the response and ensure safety notice"""
    if not response:
        return "I couldn't generate a response. Please try again."
    
    # Remove unnecessary AI disclaimers (single pass)
    response = re.sub(r"(?i)(as an ai(?: language)? model|i(?:'m| am) an? ai)", "", response)
    response = re.sub(r"\s+", " ", response).strip()
    
    # Add safety notice (shorter version)
    if "consult a healthcare professional" not in response.lower():
        response += "\n\n⚠️ Consult a healthcare professional."
    
    return response

# ===== PERIODIC MEMORY CLEANUP =====
def cleanup_memory():
    """Periodic memory cleanup"""
    try:
        gc.collect()
        if gc.garbage:
            gc.collect()
    except:
        pass

def start_cleanup_scheduler():
    """Start a background thread for periodic cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(60)  # Clean every minute
            cleanup_memory()
    
    import threading
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()

# Start the cleanup scheduler
start_cleanup_scheduler()

# ===== ROUTES =====

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_query = data.get('query')
        chat_history = data.get('history', [])

        if not user_query:
            return jsonify({'error': 'Query is missing.'}), 400

        # TRUNCATE EARLY - Limit query length
        if len(user_query) > 300:
            user_query = user_query[:300] + "..."

        # Limit chat history aggressively
        if chat_history and len(chat_history) > 2:
            chat_history = chat_history[-2:]
        
        # Normalize history with truncation
        if chat_history and isinstance(chat_history[0], dict):
            chat_history = [
                f"{m.get('sender','unknown')}: {m.get('text','')[:100]}"
                for m in chat_history
            ]

        # Detect category
        category = detect_query_category(user_query)
        video_link = None
        
        # Get video only for emergencies
        if category == 'emergency':
            video_link = get_first_aid_video(user_query)

        # Get vector store (lazy loaded)
        vector_store = get_vector_store()
        
        # Reduce number of documents retrieved
        docs = vector_store.similarity_search(user_query, k=1)  # Only 1 document
        context = "\n".join(d.page_content[:200] for d in docs) if docs else ""

        # Build prompt with limited length
        history_text = "\n".join(chat_history[-1:]) if chat_history else ""
        formatted_prompt = system_prompt.format(
            query=user_query[:200],
            history=history_text[:200],
            category=category
        )
        
        if category == 'emergency':
            formatted_prompt += "\n\n" + emergency_subprompt[:100]

        # Get LLM (lazy loaded)
        llm = get_llm()
        raw_response = llm._call(formatted_prompt)
        response = clean_response(raw_response)

        # Get professionals (limit to 1, with error handling)
        try:
            pros = find_relevant_professionals(user_query[:100], max_results=1)
            if pros:
                response += "\n\nFor specialized care, contact:"
                for p in pros:
                    response += f"\n- {p['name']} ({p['specialty']})"
        except Exception as e:
            print(f"⚠️ Professional lookup error: {e}")
            # Continue without professionals

        # Add video link if available
        if video_link:
            response += f"\n\n**First-Aid:** {video_link}"

        # Force garbage collection
        gc.collect()

        return jsonify({
            'response': response,
            'category': category,
            'video_link': video_link
        })

    except Exception as e:
        print("❌ Error in /chat endpoint:", e)
        gc.collect()
        return jsonify({
            'response': "⚠️ Please try again in a moment.",
            'error': str(e)
        }), 500

@app.route('/analyze-image', methods=['POST'])
def handle_image_analysis():
    try:
        data = request.get_json(force=True) or {}
        base64_image = data.get('image')
        
        if not base64_image:
            return jsonify({'error': 'Image is missing'}), 400
        
        # Limit image size to 3MB
        if len(base64_image) > 3 * 1024 * 1024:
            return jsonify({'error': 'Image size too large. Maximum 3MB allowed.'}), 400
        
        # Get image analyzer (lazy loaded)
        image_analyzer = get_image_analyzer()
        analysis = image_analyzer.analyze_image(base64_image, image_analysis_prompt)
        
        if not analysis:
            analysis = "Unable to analyze the image. Please try again."
        
        response_text = f"📸 **Image Analysis:**\n\n{analysis[:500]}\n\n---\n⚠️ Consult a healthcare professional."
        
        gc.collect()
        
        return jsonify({
            'response': response_text,
            'type': 'image_analysis'
        })
    
    except Exception as e:
        print("❌ Error in /analyze-image endpoint:", e)
        gc.collect()
        return jsonify({
            'error': f"Image analysis failed.",
            'type': 'error'
        }), 500

@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.get_json(force=True) or {}
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))

        if not weight or not height or weight <= 0 or height <= 0:
            return jsonify({'error': 'Valid weight and height are required.'}), 400

        bmi = weight / ((height/100) ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        interpretation = bmi_interpretation.format(
            bmi_value=round(bmi, 1),
            category=category
        )
            
        return jsonify({
            'bmi': round(bmi, 1),
            'category': category,
            'interpretation': interpretation[:300],  # Limit length
            'type': 'bmi_result'
        })
    except Exception as e:
        print("❌ Error in /calculate-bmi endpoint:", e)
        return jsonify({
            'error': f"BMI calculation error: {str(e)}",
            'type': 'error'
        }), 400

@app.route('/mood', methods=['POST'])
def mood_tracking():
    try:
        data = request.get_json(force=True) or {}
        description = data.get('description')
        mood_score = data.get('score')
        tags = data.get('tags', [])

        if not description or mood_score is None:
            return jsonify({'error': 'Missing description or mood score.'}), 400

        # Get mental health LLM (lazy loaded)
        mental_health_llm = get_mental_health_llm()
        
        prompt = mood_tracking_prompt.format(
            description=description[:150],  # Limit description length
            mood_score=mood_score,
            tags=", ".join(tags[:3])  # Limit tags
        )

        result = mental_health_llm._call(prompt)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()
        
        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += "\n\nNote: For crisis, contact a professional."

        gc.collect()
        
        return jsonify({'response': cleaned_response[:500], 'type': 'mood_tracking'})
    except Exception as e:
        print("❌ Error in mood tracking:", e)
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/cbt', methods=['POST'])
def cbt_exercises():
    try:
        data = request.get_json(force=True) or {}
        concern = data.get('concern')
        tried_strategies = data.get('tried_strategies', 'None')
        desired_outcome = data.get('desired_outcome', 'Not specified')

        if not concern:
            return jsonify({'error': 'Concern is required.'}), 400

        # Get mental health LLM (lazy loaded)
        mental_health_llm = get_mental_health_llm()
        
        prompt = cbt_exercises_prompt.format(
            concern=concern[:150],
            tried_strategies=tried_strategies[:150],
            desired_outcome=desired_outcome[:150]
        )

        result = mental_health_llm._call(prompt)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()
        
        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += "\n\nNote: Seek professional support if needed."

        gc.collect()
        
        return jsonify({'response': cleaned_response[:500], 'type': 'cbt_exercises'})
    except Exception as e:
        print("❌ Error in CBT exercises:", e)
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    memory_mb = None
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'memory_mb': memory_mb,
        'services': {
            'pinecone': 'connected' if PINECONE_API_KEY else 'disabled',
            'llm': 'available' if OPENROUTER_API_KEY else 'disabled'
        }
    })

@app.route('/memory', methods=['GET'])
def memory_check():
    """Monitor memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return jsonify({
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'memory_percent': round(process.memory_percent(), 2),
            'cpu_percent': round(process.cpu_percent(), 2)
        })
    except:
        return jsonify({'error': 'Memory monitoring not available'}), 500

@app.route('/')
def root():
    return 'OK', 200

if __name__ == '__main__':
    # Force garbage collection before starting
    gc.collect()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=False, 
        host='0.0.0.0', 
        port=port,
        threaded=False,  # Disable threading to reduce memory
        processes=1      # Single process
    )
