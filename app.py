from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
from pinecone import Pinecone, data
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from langchain_pinecone import Pinecone as LangchainPinecone
import re
import requests
from langchain.llms.base import LLM
from pydantic import Field
from transformers import pipeline

app = Flask(__name__)

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Monkey-patch pinecone.Index to use the new client’s index type
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
        # Use OpenRouter's endpoint for chat completions
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek/deepseek-chat:free",  # Ensure this matches your account tier on OpenRouter
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

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']

    # Retrieve relevant documents from Pinecone
    try:
        docs = vector_store.similarity_search(user_query, k=5)  # Retrieve top 5 relevant documents
        print(f"Retrieved {len(docs)} documents from Pinecone.")
        if not docs:
            return jsonify({'response': 'I do not know.'})
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Context: {context}")

        # Skip summarization to retain detailed context
        # context = summarize_context(context)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return jsonify({'response': f"Error retrieving data: {str(e)}"}), 500

    # Generate response using Deepseek LLM
    try:
        formatted_prompt = system_prompt.format(context=context)
        print(f"Formatted Prompt: {formatted_prompt}")

        # Call the Deepseek LLM to generate a response
        raw_response = llm._call(formatted_prompt)
        print(f"Raw Response: {raw_response}")

        # Clean the response
        response = clean_response(raw_response)
        print(f"Cleaned Response: {response}")
    except Exception as e:
        print(f"Error during model generation: {e}")
        response = "I encountered an error while processing your request."

    # Return the response as JSON
    return jsonify({'response': response})

def clean_response(response):
    """Clean up the generated response."""
    # Remove irrelevant phrases
    response = re.sub(r"(encyclopedia|names are not related|experimental approach)", "", response, flags=re.IGNORECASE)

    # Ensure the response ends with punctuation (e.g., period, exclamation mark)
    if not response.endswith(('.', '?', '!')):
        # Only trim incomplete sentences if they are very short (less than 5 characters)
        response = re.sub(r"[^\.!?]{1,5}$", "", response)

    return response.strip()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))