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
from pymongo import MongoClient
import nltk

# ===== NLTK DATA SETUP =====
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
Path(NLTK_DATA_PATH).mkdir(parents=True, exist_ok=True)

try:
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('wordnet', download_dir=NLTK_DATA_PATH)
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"❌ NLTK download error: {str(e)}")

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

# Load environment variables (you already have load_dotenv earlier)
MONGODB_URI    = os.environ.get("MONGODB_URI")
MONGODB_DB     = os.environ.get("MONGODB_DB_NAME", "medi_bot")

mongo_client   = MongoClient(MONGODB_URI)
db             = mongo_client[MONGODB_DB]
symptoms_col   = db["symptoms"]

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

# Connect to existing Pinecone index
vector_store = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Define the Deepseek LLM class
class DeepseekLLM(LLM):
    api_key: str = Field(..., description="OpenRouter API key")
    temperature: float = Field(0.4, description="Sampling temperature")
    max_tokens: int = Field(2000, description="Maximum number of tokens to generate")
    
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
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

# Initialize the Deepseek LLM
llm = DeepseekLLM(api_key=OPENROUTER_API_KEY, temperature=0.4, max_tokens=2000)

class QwenImageAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_image(self, base64_image, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": "qwen/qwen2.5-vl-72b-instruct:free",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "temperature": 0.4,
            "max_tokens": 1000
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

# Initialize Qwen analyzer
image_analyzer = QwenImageAnalyzer(OPENROUTER_API_KEY)

# Emergency keywords detection
EMERGENCY_KEYWORDS = {
    'faint', 'seizure', 'choking', 'bleeding', 'stroke', 'nosebleed',
    'unconscious', 'heart attack', 'overdose', 'cardiac arrest', 'no pulse',
    'not breathing', 'gasping', 'CPR', 'AED', 'chest pain', 'tight chest',
    'unresponsive', 'dizzy', 'confused', 'slurred speech', 'numbness', 'paralyzed',
    'broken bone', 'fracture', 'head injury', 'concussion', 'burn', 'cut', 'sprain',
    'deep wound', 'heavy bleeding', 'open fracture', 'hypothermia', 'heat-stroke',
    'frostbite', 'dehydration', 'drowning', 'allergic reaction', 'anaphylaxis',
    'epipen', 'bee sting', 'snake bite', 'poisoning', 'infant not breathing',
    'baby choking', 'child unresponsive'
}


def detect_query_category(query):
    """Determine if query is emergency or normal"""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS):
        return "emergency"
    return "non-emergency"

# === Kimi LLM for Mental Health ===
class KimiLLM(LLM):
    api_key: str = Field(..., description="OpenRouter API key")
    temperature: float = Field(0.4, description="Sampling temperature")
    max_tokens: int = Field(1500, description="Maximum tokens")

    @property
    def _llm_type(self) -> str:
        return "kimi-vl-a3b-thinking"

    def _call(self, prompt: str, stop=None) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "moonshotai/kimi-vl-a3b-thinking:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# Initialize Kimi LLM
kimi_llm = KimiLLM(api_key=OPENROUTER_API_KEY, temperature=0.4, max_tokens=1500)

# Mock wearable data integration
def get_wearable_data():
    """Simulate fetching wearable device data"""
    return {
        "heart_rate": 72,
        "blood_pressure": "120/80",
        "temperature": 98.6,
        "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "oxygen_saturation": 98
    }

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_query = data.get('query')
        chat_history = data.get('history', [])

        if not user_query:
            return jsonify({'error': 'Query is missing.'}), 400

        # Normalize history into “sender: text” strings
        if chat_history and isinstance(chat_history[0], dict):
            chat_history = [
                f"{m.get('sender','unknown')}: {m.get('text','')}"
                for m in chat_history
            ]

        # Detect emergency vs. normal
        category = detect_query_category(user_query)
        video_link = None
        if category == 'emergency':
            video_link = get_first_aid_video(user_query)

        # Retrieve context from Pinecone
        docs = vector_store.similarity_search(user_query, k=3)
        context = "\n".join(d.page_content for d in docs) if docs else ""

        # Build and call the LLM prompt
        formatted_prompt = system_prompt.format(
            query=user_query,
            history="\n".join(chat_history[-3:]),
            category=category
        )
        if category == 'emergency':
            formatted_prompt += "\n\n" + emergency_subprompt

        raw_response = llm._call(formatted_prompt)
        response = clean_response(raw_response)

        # Append specialists if any
        pros = find_relevant_professionals(user_query, max_results=2)
        if pros:
            response += "\n\nFor specialized care at Metro Hospital, you may contact:"
            for p in pros:
                response += (
                    f"\n- {p['name']} ({p['specialty']}), 📞 {p['phone']}, ✉️ {p['email']}"
                )

        # ★ Inject the first‑aid video link into the response text ★
        if video_link:
            response += f"\n\n**First‑Aid Video:** {video_link}"

        return jsonify({
            'response':     response,
            'category':     category,
            'video_link':   video_link,
            'context_used': [d.page_content[:100] + "..." for d in docs]
        })

    except Exception as e:
        print("❌ Error in /chat endpoint:", e)
        return jsonify({
            'response': "⚠️ Medical analysis error. Please try again or consult a doctor directly.",
            'error':    str(e)
        }), 500


@app.route('/analyze-image', methods=['POST'])
def handle_image_analysis():
    try:
        data = request.get_json(force=True) or {}
        base64_image = data.get('image')
        if not base64_image:
            return jsonify({'error': 'Image is missing'}), 400
        
        analysis = image_analyzer.analyze_image(base64_image, image_analysis_prompt)
        return jsonify({
            'response': f"IMAGE ANALYSIS:\n{analysis}\n\nNOTE: Always verify with a healthcare professional",
            'type': 'image_analysis'
        })
    
    except Exception as e:
        print("❌ Error in /analyze-image endpoint:", e)
        return jsonify({
            'error': f"Image analysis failed: {str(e)}",
            'type': 'error'
        }), 500

@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.get_json(force=True) or {}
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))

        if not weight or not height:
            return jsonify({'error': 'Weight or height is missing.'}), 400

        bmi = weight / ((height/100) ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        # Format BMI interpretation using the prompt template
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

# Mental health endpoints
@app.route('/mood', methods=['POST'])
def mood_tracking():
   

    data = request.get_json(force=True) or {}

    description = data.get('description')
    mood_score = data.get('score')
    tags = data.get('tags', [])

    if not description or mood_score is None:
        return jsonify({'error': 'Missing description or mood score.'}), 400

    try:
        # Format the mood prompt using structured input
        prompt = mood_tracking_prompt.format(
            description=description,
            mood_score=mood_score,
            tags=", ".join(tags)
        )

        result = kimi_llm._call(prompt)

        # Remove internal monologue between ◁think▷ and ◁/think▷ (including tags)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()

        return jsonify({'response': cleaned_response, 'type': 'mood_tracking'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cbt', methods=['POST'])
def cbt_exercises():
    data = request.get_json(force=True) or {}

    concern = data.get('concern')
    tried_strategies = data.get('tried_strategies', 'None')
    desired_outcome = data.get('desired_outcome', 'Not specified')

    if not concern:
        return jsonify({'error': 'Concern is required.'}), 400

    try:
        # Format the CBT prompt using structured input
        prompt = cbt_exercises_prompt.format(
            concern=concern,
            tried_strategies=tried_strategies,
            desired_outcome=desired_outcome
        )

        result = kimi_llm._call(prompt)

        # Remove internal monologue between ◁think▷ and ◁/think▷
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()

        return jsonify({'response': cleaned_response, 'type': 'cbt_exercises'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def clean_response(response):
    # Remove unnecessary disclaimers while keeping safety notices
    response = re.sub(r"(as an ai(?: language)? model|i(?:'m| am) an? ai)", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\s+", " ", response).strip()
    
    # Ensure safety notice is present
    if "consult a healthcare professional" not in response.lower():
        response += "\n\nNote: Consult a healthcare professional for medical advice"
    
    return response


@app.route('/symptoms', methods=['POST'])
def submit_symptom():
    data = request.get_json(force=True) or {}

    # 🛡️ Basic validation
    required = ["date", "symptom", "severity"]
    if not all(field in data and data[field] for field in required):
        return jsonify({'error': 'Missing one of: date, symptom, severity'}), 400

    # Build the document
    doc = {
        "date":        data["date"],
        "symptom":     data["symptom"],
        "severity":    data["severity"],
        "duration":    data.get("duration"),
        "triggers":    data.get("triggers", []),
        "medications": data.get("medications"),
        "notes":       data.get("notes"),
        "created_at":  datetime.utcnow()
    }

    try:
        result = symptoms_col.insert_one(doc)
        return jsonify({
            'status':      'ok',
            'inserted_id': str(result.inserted_id)
        }), 200

    except Exception as e:
        print("❌ Error inserting symptom:", e)
        return jsonify({'error': str(e)}), 500



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'pinecone': index.describe_index_stats() if PINECONE_API_KEY else 'disabled',
            'llm': 'deepseek' if OPENROUTER_API_KEY else 'disabled'
        }
    })

if __name__ == '__main__':
    port = int(os.environ['PORT'])  # No default value; Render sets this
    app.run(debug=False, host='0.0.0.0', port=port)

