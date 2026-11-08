"""MediBot Flask application (modernized).

Endpoints and response shapes are unchanged from the legacy app:
  POST /chat           -> {response, category, video_link}
  POST /analyze-image  -> {response, type}
  POST /calculate-bmi  -> {bmi, category, interpretation, type}
  POST /mood           -> {response, type}
  POST /cbt            -> {response, type}
  GET  /health         -> {status, memory_mb, services}
  GET  /memory         -> {memory_mb, memory_percent, cpu_percent}
  GET  /               -> 'OK'

Heavy resources (embedding service, Pinecone, LLM failover layer, agents) are
initialized lazily and reused across requests, and the app is Gunicorn-ready.

A startup warmup loads the embedding model and builds the coordinator so the
first /chat request does not pay the cold-start cost.
"""
import logging
import re
import time
from threading import Lock

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import OPENROUTER_API_KEY, PINECONE_API_KEY, VALIDATION_ENABLED
from src.prompt import (
    bmi_interpretation,
    image_analysis_prompt,
)
from video_link import get_first_aid_video  # noqa: F401  (used via coordinator; re-exported for compatibility)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://medibot-frontend-eight.vercel.app",
])

# ==========================================================================
# Lazily-initialized shared services (created once per process)
# ==========================================================================
_coordinator = None
_coordinator_lock = Lock()
_init_error = None
_warmup_done = False


def _build_coordinator():
    """Import and build the agent coordinator once; record init errors."""
    global _coordinator, _init_error
    if _coordinator is None and _init_error is None:
        with _coordinator_lock:
            if _coordinator is None and _init_error is None:
                try:
                    from src.agent_coordinator import AgentCoordinator

                    _coordinator = AgentCoordinator()
                    logger.info("Agent coordinator initialized.")
                except Exception as exc:
                    # Do not crash the whole app at import time: /health must stay
                    # available so Render can report a meaningful state.
                    _init_error = str(exc)
                    logger.exception("Coordinator initialization failed: %s", exc)
    return _coordinator


def get_coordinator():
    return _build_coordinator()


def _warm_services() -> None:
    """Pre-load heavy resources so the first /chat request is fast.

    Runs once per process. Failures are logged but never fatal — the app will
    simply cold-start on the first real request instead.
    """
    global _warmup_done
    if _warmup_done:
        return
    start = time.time()

    # 1. Load the embedding model and force a single inference.
    try:
        from src.embedding_service import get_embedding_service
        svc = get_embedding_service()
        logger.info("Warming embedding service (provider=%s, model=%s, dim=%d)...",
                    svc.provider_tag, svc.model_name, svc.dimension)
        t0 = time.time()
        vec = svc.embed_query("warmup")
        logger.info("Embedding warmup complete in %.1fs (%d dims).", time.time() - t0, len(vec))
    except Exception as exc:
        logger.warning("Embedding warmup failed (will cold-start on first request): %s", exc)

    # 2. Build the coordinator (LangGraph, router, specialist agents).
    try:
        if _build_coordinator() is None:
            logger.warning("Coordinator not built during warmup (see earlier logs).")
    except Exception as exc:
        logger.warning("Coordinator warmup failed: %s", exc)

    _warmup_done = True
    logger.info("Service warmup finished in %.1fs.", time.time() - start)


# Run warmup once at import time so both `python app.py` and Gunicorn benefit.
# Under Gunicorn this executes in each worker process after fork, which is what
# we want: each worker pre-loads its own model instance.
_warm_services()


# ==========================================================================
# /chat - main chat endpoint (agent coordinator)
# ==========================================================================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_query = data.get('query')
        chat_history = data.get('history', [])

        if not user_query:
            return jsonify({'error': 'Query is missing.'}), 400

        # Limit chat history to save memory (legacy behaviour).
        if chat_history and len(chat_history) > 3:
            chat_history = chat_history[-3:]

        coordinator = _build_coordinator()
        if coordinator is None:
            return jsonify({
                'response': "The assistant is still starting up. Please try again in a moment.",
                'error': _init_error or "coordinator_unavailable",
                'category': 'error',
                'video_link': None,
            }), 503

        result = coordinator.handle(user_query, history=chat_history)
        return jsonify(result)

    except Exception as exc:
        logger.exception("Error in /chat endpoint: %s", exc)
        return jsonify({
            'response': "⚠️ I'm experiencing high load. Please try again in a moment.",
            'error': str(exc),
        }), 500


# ==========================================================================
# /analyze-image - OpenRouter vision model
# ==========================================================================
@app.route('/analyze-image', methods=['POST'])
def handle_image_analysis():
    try:
        data = request.get_json(force=True) or {}
        base64_image = data.get('image')

        if not base64_image:
            return jsonify({'error': 'Image is missing'}), 400

        if len(base64_image) > 5 * 1024 * 1024:  # 5MB limit (legacy behaviour)
            return jsonify({'error': 'Image size too large. Maximum 5MB allowed.'}), 400

        from src.llm_manager import analyze_image as run_image_analysis
        analysis = run_image_analysis(base64_image, image_analysis_prompt)

        if not analysis:
            analysis = "Unable to analyze the image. Please try again."

        response_text = (
            f"📸 **Image Analysis Results:**\n\n{analysis}\n\n---\n"
            "⚠️ **Important Note:** This analysis is for informational purposes only. "
            "Always consult with a qualified healthcare professional for medical advice, "
            "diagnosis, or treatment."
        )

        return jsonify({'response': response_text, 'type': 'image_analysis'})

    except Exception as exc:
        logger.exception("Error in /analyze-image endpoint: %s", exc)
        return jsonify({
            'error': "Image analysis failed. Please try again.",
            'type': 'error',
        }), 500


# ==========================================================================
# /calculate-bmi - pure computation, no external services
# ==========================================================================
@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.get_json(force=True) or {}
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))

        if not weight or not height or weight <= 0 or height <= 0:
            return jsonify({'error': 'Valid weight and height are required.'}), 400

        bmi = weight / ((height / 100) ** 2)

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
            category=category,
        )

        return jsonify({
            'bmi': round(bmi, 1),
            'category': category,
            'interpretation': interpretation,
            'type': 'bmi_result',
        })
    except Exception as exc:
        logger.exception("Error in /calculate-bmi endpoint: %s", exc)
        return jsonify({
            'error': f"BMI calculation error: {str(exc)}",
            'type': 'error',
        }), 400


# ==========================================================================
# /mood and /cbt - Mental Health Agent
# ==========================================================================
@app.route('/mood', methods=['POST'])
def mood_tracking():
    try:
        data = request.get_json(force=True) or {}
        description = data.get('description')
        mood_score = data.get('score')
        tags = data.get('tags', [])

        if not description or mood_score is None:
            return jsonify({'error': 'Missing description or mood score.'}), 400

        from src.agents.mental_health_agent import MentalHealthAgent
        from src.llm_manager import LLMError

        agent = MentalHealthAgent()
        result = agent.mood_tracking(description, mood_score, tags)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()

        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += ("\n\nNote: This is for mental wellness support. For crisis situations, "
                                 "please contact a mental health professional or emergency services.")

        return jsonify({'response': cleaned_response, 'type': 'mood_tracking'})
    except LLMError as exc:
        logger.error("LLM failure in mood tracking: %s", exc)
        return jsonify({'error': "The mental health assistant is temporarily unavailable. "
                                 "For crisis support, call or text 988."}), 503
    except Exception as exc:
        logger.exception("Error in mood tracking: %s", exc)
        return jsonify({'error': str(exc)}), 500


@app.route('/cbt', methods=['POST'])
def cbt_exercises():
    try:
        data = request.get_json(force=True) or {}
        concern = data.get('concern')
        tried_strategies = data.get('tried_strategies', 'None')
        desired_outcome = data.get('desired_outcome', 'Not specified')

        if not concern:
            return jsonify({'error': 'Concern is required.'}), 400

        from src.agents.mental_health_agent import MentalHealthAgent
        from src.llm_manager import LLMError

        agent = MentalHealthAgent()
        result = agent.cbt_exercises(concern, tried_strategies, desired_outcome)
        cleaned_response = re.sub(r'◁think▷.*?◁/think▷', '', result, flags=re.DOTALL).strip()

        if "consult a healthcare professional" not in cleaned_response.lower():
            cleaned_response += ("\n\nNote: These exercises are for self-help purposes. If you're "
                                 "experiencing severe distress, please seek professional support.")

        return jsonify({'response': cleaned_response, 'type': 'cbt_exercises'})
    except LLMError as exc:
        logger.error("LLM failure in CBT exercises: %s", exc)
        return jsonify({'error': "The mental health assistant is temporarily unavailable. "
                                 "For crisis support, call or text 988."}), 503
    except Exception as exc:
        logger.exception("Error in CBT exercises: %s", exc)
        return jsonify({'error': str(exc)}), 500


# ==========================================================================
# Monitoring endpoints
# ==========================================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    memory_mb = None
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        pass

    services = {
        'pinecone': 'connected' if PINECONE_API_KEY else 'disabled',
        'llm': 'available' if OPENROUTER_API_KEY else 'disabled',
        'coordinator': 'ready' if _coordinator is not None else ('error' if _init_error else 'lazy'),
        'validation': 'llm' if VALIDATION_ENABLED else 'rules',
        'warmup': 'done' if _warmup_done else 'pending',
    }
    return jsonify({'status': 'healthy', 'memory_mb': memory_mb, 'services': services})


@app.route('/memory', methods=['GET'])
def memory_check():
    """Monitor memory usage."""
    try:
        import psutil
        process = psutil.Process()
        return jsonify({
            'memory_mb': round(process.memory_info().rss / 1024 / 1024, 2),
            'memory_percent': round(process.memory_percent(), 2),
            'cpu_percent': round(process.cpu_percent(), 2),
        })
    except Exception:
        return jsonify({'error': 'Memory monitoring not available'}), 500


@app.route('/')
def root():
    return 'OK', 200


if __name__ == '__main__':
    from src.config import PORT

    app.run(debug=False, host='0.0.0.0', port=PORT)