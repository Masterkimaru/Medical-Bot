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
import nltk
import gc
import time
from functools import wraps
from threading import Lock

# ===== MEMORY OPTIMIZATION =====
# Force garbage collection on startup
gc.collect()

# ===== NLTK DATA SETUP (Optimized) =====
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
Path(NLTK_DATA_PATH).mkdir(parents=True, exist_ok=True)

def setup_nltk():
    """Lazy load NLTK data only if needed"""
    try:
        # Check if data already exists
        from nltk.data import find
        try:
            find('tokenizers/punkt')
            print("✅ NLTK data already exists")
            return True
        except LookupError:
            pass
            
        # Download only what's needed with quiet mode
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('wordnet', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ NLTK setup warning: {str(e)}")
        return False

# Call this after imports
setup_nltk()

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
    temperature: float = Field(0.3, description="Sampling temperature")  # Reduced
    max_tokens: int = Field(1000, description="Maximum number of tokens to generate")  # Reduced from 2000
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop=None) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=25)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
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
                    temperature=0.3, 
                    max_tokens=1000
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
            "model": "nvidia/nemotron-3.5-content-safety:free",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "temperature": 0.3,
            "max_tokens": 500  # Reduced from 1000
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=20)
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
    temperature: float = Field(0.3, description="Sampling temperature")  # Reduced
    max_tokens: int = Field(800, description="Maximum tokens")  # Reduced from 1500
    site_url: str = Field("", description="Optional site URL for rankings")
    site_name: str = Field("", description="Optional site title for rankings")

    @property
    def _llm_type(self) -> str:
        return "qwen-next-80b"

    def _call(self, prompt: str, stop=None) -> str:
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
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
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
                    temperature=0.3, 
                    max_tokens=800,
                    site_url="https://your-site-url.com",
                    site_name="MediBot"
                )
    return _mental_health_llm

# Clean response function (optimized)
def clean_response(response):
    """Clean the response and ensure safety notice"""
    if not response:
        return "I couldn't generate a response. Please try again."
    
    # Remove unnecessary AI disclaimers
    response = re.sub(r"(as an ai(?: language)? model|i(?:'m| am) an? ai)", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\s+", " ", response).strip()
    
    # Add safety notice if not present
    if "consult a healthcare professional" not in response.lower():
        response += "\n\n⚠️ Note: Consult a healthcare professional for medical advice"
    
    return response

# ===== ROUTES =====

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_query = data.get('query')
        chat_history = data.get('history', [])

        if not user_query:
            return jsonify({'error': 'Query is missing.'}), 400

        # Limit chat history to save memory
        if chat_history and len(chat_history) > 3:
            chat_history = chat_history[-3:]
        
        # Normalize history
        if chat_history and isinstance(chat_history[0], dict):
            chat_history = [
                f"{m.get('sender','unknown')}: {m.get('text','')}"
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
        docs = vector_store.similarity_search(user_query, k=2)  # Reduced from 3
        context = "\n".join(d.page_content for d in docs) if docs else ""

        # Build prompt
        history_text = "\n".join(chat_history[-2:]) if chat_history else ""  # Limit history
        formatted_prompt = system_prompt.format(
            query=user_query,
            history=history_text,
            category=category
        )
        
        if category == 'emergency':
            formatted_prompt += "\n\n" + emergency_subprompt

        # Get LLM (lazy loaded)
        llm = get_llm()
        raw_response = llm._call(formatted_prompt)
        response = clean_response(raw_response)

        # Get professionals (limit to 1 for memory)
        pros = find_relevant_professionals(user_query, max_results=1)  # Reduced from 2
        if pros:
            response += "\n\nFor specialized care at Metro Hospital, you may contact:"
            for p in pros:
                response += f"\n- {p['name']} ({p['specialty']}), 📞 {p['phone']}"

        # Add video link if available
        if video_link:
            response += f"\n\n**First‑Aid Video:** {video_link}"

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
            'response': "⚠️ I'm experiencing high load. Please try again in a moment.",
            'error': str(e)
        }), 500

@app.route('/analyze-image', methods=['POST'])
def handle_image_analysis():
    try:
        data = request.get_json(force=True) or {}
        base64_image = data.get('image')
        
        if not base64_image:
            return jsonify({'error': 'Image is missing'}), 400
        
        # Limit image size
        if len(base64_image) > 5 * 1024 * 1024:  # Reduced to 5MB
            return jsonify({'error': 'Image size too large. Maximum 5MB allowed.'}), 400
        
        # Get image analyzer (lazy loaded)
        image_analyzer = get_image_analyzer()
        analysis = image_analyzer.analyze_image(base64_image, image_analysis_prompt)
        
        if not analysis:
            analysis = "Unable to analyze the image. Please try again."
        
        response_text = f"📸 **Image Analysis Results:**\n\n{analysis}\n\n---\n⚠️ **Important Note:** This analysis is for informational purposes only. Always consult with a qualified healthcare professional for medical advice, diagnosis, or treatment."
        
        gc.collect()
        
        return jsonify({
            'response': response_text,
            'type': 'image_analysis'
        })
    
    except Exception as e:
        print("❌ Error in /analyze-image endpoint:", e)
        gc.collect()
        return jsonify({
            'error': f"Image analysis failed. Please try again.",
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
            'interpretation': interpretation,
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
            description=description[:200],  # Limit description length
            mood_score=mood_score,
            tags=", ".join(tags[:5])  # Limit tags
        )

        result = mental_health_llm._call(prompt)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()
        
        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += "\n\nNote: This is for mental wellness support. For crisis situations, please contact a mental health professional or emergency services."

        gc.collect()
        
        return jsonify({'response': cleaned_response, 'type': 'mood_tracking'})
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
            concern=concern[:200],  # Limit concern length
            tried_strategies=tried_strategies[:200],
            desired_outcome=desired_outcome[:200]
        )

        result = mental_health_llm._call(prompt)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()
        
        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += "\n\nNote: These exercises are for self-help purposes. If you're experiencing severe distress, please seek professional support."

        gc.collect()
        
        return jsonify({'response': cleaned_response, 'type': 'cbt_exercises'})
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
    port = int(os.environ.get('PORT', 5000))  # Added default
    app.run(debug=False, host='0.0.0.0', port=port)
