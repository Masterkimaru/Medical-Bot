# 🩺 MediBot — AI Medical Assistant

MediBot is a full-stack AI-powered medical assistant that answers health questions, analyzes medical images, calculates BMI, tracks mood, and provides CBT-based mental health exercises. It combines **retrieval-augmented generation (RAG)** over a medical knowledge base with **large language models** to deliver contextual, safety-conscious responses.

> ⚠️ **Medical Disclaimer:** MediBot is for educational and informational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional. In an emergency, contact local emergency services immediately.

---

## ✨ Features

- 💬 **Medical Chat (RAG)** — Ask health questions; answers are grounded in a Pinecone vector store of medical knowledge (`/chat`)
- 🚨 **Emergency Detection** — Automatically detects emergency keywords (e.g., *choking, cardiac arrest, seizure, anaphylaxis*) and appends first-aid guidance + instructional video links
- 📸 **Medical Image Analysis** — Upload an image and get an AI-generated analysis (`/analyze-image`)
- ⚖️ **BMI Calculator** — Instant BMI computation with category and interpretation (`/calculate-bmi`)
- 🧠 **Mood Tracking** — Describe your feelings and get supportive, reflective feedback (`/mood`)
- 🛋️ **CBT Exercises** — Cognitive Behavioral Therapy-style exercises tailored to your concern (`/cbt`)
- 👨‍⚕️ **Professional Referrals** — Suggests relevant on-site healthcare professionals when appropriate
- 📊 **Health & Memory Monitoring** — Built-in endpoints for uptime and resource monitoring

---

## 🧰 Tech Stack

| Layer | Technology |
| --- | --- |
| **Frontend** | React, TypeScript, Vite |
| **Backend** | Python, Flask, Flask-CORS |
| **LLM Provider** | OpenRouter API |
| **General & Vision Model** | `nvidia/nemotron-3.5-lightning:free` |
| **Mental Health Model** | `ainclusionai/ling-3.0-flash:free` |
| **Vector Database** | Pinecone (`doctorbot` index) |
| **Embeddings** | HuggingFace sentence-transformers (via LangChain) |
| **Orchestration** | LangChain (`langchain-pinecone`) |
| **NLP Utilities** | NLTK (lazy-loaded) |

---

## 🗂️ Repositories

This project is split into two repositories:

| Repo | Description |
| --- | --- |
| 🔧 [`Medical-Bot`](https://github.com/Masterkimaru/Medical-Bot) | Flask backend API (this `app.py`) |
| 🎨 [`Medibot-Frontend`](https://github.com/Masterkimaru/Medibot-Frontend) | React + TypeScript + Vite frontend |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- API keys:
- [OpenRouter API key](https://openrouter.ai/keys)
- [Pinecone API key](https://app.pinecone.io/) with an index named `doctorbot`

---

### 1️⃣ Clone the Repositories

```bash
git clone https://github.com/Masterkimaru/Medical-Bot.git
git clone https://github.com/Masterkimaru/Medibot-Frontend.git
```

---

### 2️⃣ Run the Backend

```bash
cd Medical-Bot

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env   # then fill in your keys (see table below)
```

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
SITE_URL=https://your-site-url.com
SITE_NAME=MediBot
PORT=5000
```

> **NLTK data** is downloaded lazily on first use to `/opt/render/nltk_data` by default (override with the `NLTK_DATA` environment variable if running locally).

Start the server:

```bash
python app.py
```

The API will be available at **`http://localhost:5000`**.

---

### 3️⃣ Run the Frontend

In a new terminal:

```bash
cd Medibot-Frontend

npm install
npm run dev
```

The app will be available at **`http://localhost:5173`** — the backend CORS is already pre-configured for this origin, plus the production Vercel deployment.

For a production build:

```bash
npm run build
npm run preview
```

---

## 🔌 API Reference

Base URL: `http://localhost:5000`

| Method | Endpoint | Description | Key Body Fields |
| --- | --- | --- | --- |
| `POST` | `/chat` | Medical Q&A with RAG + emergency detection | `query`, `history` |
| `POST` | `/analyze-image` | AI analysis of a medical image (base64 JPEG) | `image` |
| `POST` | `/calculate-bmi` | BMI calculation and interpretation | `weight` (kg), `height` (cm) |
| `POST` | `/mood` | Mood tracking and reflection | `description`, `score`, `tags` |
| `POST` | `/cbt` | CBT-style exercises | `concern`, `tried_strategies`, `desired_outcome` |
| `GET` | `/health` | Service health & memory status | — |
| `GET` | `/memory` | Live memory/CPU usage | — |
| `GET` | `/` | Liveness check (`OK`) | — |

### Example: Chat Request

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of dehydration?", "history": []}'
```

```json
{
  "response": "Common symptoms of dehydration include... ⚠️ Consult a healthcare professional.",
  "category": "non-emergency",
  "video_link": null
}
```

---

## 🌍 Deployment

The backend is optimized for memory-constrained hosts (e.g., **Render free tier**):

- 512MB address-space limit (`RLIMIT_AS`)
- Lazy loading of LLM, vector store, and image analyzer instances
- Aggressive garbage collection with a background cleanup scheduler
- Response caching with a bounded cache size
- Single-process, non-threaded Flask configuration
- Requests timeouts capped at 15s; image uploads limited to 3MB

**Frontend** is deployed on **Vercel** (`https://medibot-frontend-eight.vercel.app`) — make sure the backend's `CORS` origins list includes your frontend URL.

---

## 🔐 Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter API key (used for all LLM calls) |
| `SITE_URL` | ➖ | Sent as `HTTP-Referer` header to OpenRouter |
| `SITE_NAME` | ➖ | Sent as `X-OpenRouter-Title` header (default: `MediBot`) |
| `PORT` | ➖ | Server port (default: `5000`) |
| `NLTK_DATA` | ➖ | NLTK data directory (default: `/opt/render/nltk_data`) |

---

## ⚠️ Security & Privacy

- Never commit your `.env` file — it is listed in `.gitignore`.
- Avoid sending personally identifiable health information through the chat in shared/public deployments.
- Review the data retention policies of third-party APIs (OpenRouter, Pinecone) before handling real user data.

---

## 📝 License

This project is open for educational use. Add a `LICENSE` file (e.g., MIT) to specify terms.

---

## 👤 Author

**Kelvin Wachira**

- GitHub: [@Masterkimaru](https://github.com/Masterkimaru)
- LinkedIn: [kelvin-wachira-51b2902a0](https://linkedin.com/in/kelvin-wachira-51b2902a0)
