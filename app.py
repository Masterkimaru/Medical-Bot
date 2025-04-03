from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
import re
import requests
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

# Import prompt definitions from src/prompt.py
from src.prompt import system_prompt, image_analysis_prompt

from pinecone import Pinecone, data
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.llms.base import LLM
from pydantic import Field

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data['query']
        chat_history = data.get('history', [])

        # Retrieve relevant medical context
        docs = vector_store.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs]) if docs else "No specific medical context available"

        # Generate response with medical workflow using the imported system_prompt
        formatted_prompt = system_prompt.format(
            context=context,
            query=user_query,
            history="\n".join(chat_history[-3:]),  # Last 3 exchanges
            wearable_data=get_wearable_data()
        )

        raw_response = llm._call(formatted_prompt)
        response = clean_response(raw_response)

        return jsonify({
            'response': response,
            'context_used': [doc.page_content[:100] + "..." for doc in docs]  # For debugging
        })

    except Exception as e:
        return jsonify({
            'response': "⚠️ Medical analysis error. Please try again or consult a doctor directly.",
            'error': str(e)
        }), 500

@app.route('/analyze-image', methods=['POST'])
def handle_image_analysis():
    try:
        data = request.json
        base64_image = data['image']
        
        # Use the imported image_analysis_prompt instead of redefining it
        analysis = image_analyzer.analyze_image(base64_image, image_analysis_prompt)
        return jsonify({
            'response': f"IMAGE ANALYSIS:\n{analysis}\n\nNOTE: Always verify with a healthcare professional",
            'type': 'image_analysis'
        })
    
    except Exception as e:
        return jsonify({
            'error': f"Image analysis failed: {str(e)}",
            'type': 'error'
        }), 500

@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.json
        weight = float(data['weight'])
        height = float(data['height'])
        
        bmi = weight / ((height/100) ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        return jsonify({
            'bmi': round(bmi, 1),
            'category': category,
            'interpretation': get_bmi_interpretation(category),
            'type': 'bmi_result'
        })
    except Exception as e:
        return jsonify({
            'error': f"BMI calculation error: {str(e)}",
            'type': 'error'
        }), 400

def get_bmi_interpretation(category):
    # You can use the imported bmi_interpretation as reference or logging if needed.
    interpretations = {
        "Underweight": "May indicate nutritional deficiency or other conditions",
        "Normal weight": "Healthy weight range for your height",
        "Overweight": "Increased risk for health conditions",
        "Obese": "High risk for serious health conditions"
    }
    return interpretations.get(category, "Consult a healthcare provider for personalized advice")

def clean_response(response):
    # Remove unnecessary disclaimers while keeping safety notices
    response = re.sub(r"(as an ai(?: language)? model|i(?:'m| am) an? ai)", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\s+", " ", response).strip()
    
    # Ensure safety notice is present
    if "consult a healthcare professional" not in response.lower():
        response += "\n\nNote: Consult a healthcare professional for medical advice"
    
    return response

if __name__ == '__main__':
    # On Render, PORT is guaranteed to be set.
    port = int(os.environ['PORT'])
    app.run(debug=False, host='0.0.0.0', port=port)
