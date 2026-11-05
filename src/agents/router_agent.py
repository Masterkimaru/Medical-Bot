"""Emergency + category router.

Deterministic keyword screening first (preserves the legacy behaviour and
latency profile); the LLM is only used as a refinement pass when the query is
ambiguous (neither clearly emergency nor clearly topic-scoped).
"""
import logging
import re
from typing import Dict

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# Ported verbatim from the legacy app.py so behaviour is unchanged.
EMERGENCY_KEYWORDS = {
    'faint', 'seizure', 'choking', 'bleeding', 'stroke', 'nosebleed',
    'unconscious', 'heart attack', 'overdose', 'cardiac arrest', 'no pulse',
    'not breathing', 'gasping', 'cpr', 'aed', 'chest pain', 'tight chest',
    'unresponsive', 'dizzy', 'confused', 'slurred speech', 'numbness', 'paralyzed',
    'broken bone', 'fracture', 'head injury', 'concussion', 'burn', 'cut', 'sprain',
    'deep wound', 'heavy bleeding', 'open fracture', 'hypothermia', 'heat-stroke',
    'frostbite', 'dehydration', 'drowning', 'allergic reaction', 'anaphylaxis',
    'epipen', 'bee sting', 'snake bite', 'poisoning',
}

MENTAL_HEALTH_KEYWORDS = {
    'depressed', 'depression', 'anxious', 'anxiety', 'panic attack', 'mood',
    'suicidal', 'self harm', 'self-harm', 'stress', 'burnout', 'sad', 'lonely',
    'mental health', 'cbt', 'hopeless',
}

MEDICATION_KEYWORDS = {
    'dose', 'dosage', 'mg', 'ml', 'tablet', 'tablets', 'capsule', 'capsules',
    'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'antibiotic',
    'antibiotics', 'prescription', 'medicine', 'medication', 'drug', 'insulin',
    'inhaler', 'side effects', 'overdose',
}


class RouterAgent(BaseAgent):
    name = "router"
    role = "medical"

    def classify(self, query: str) -> Dict[str, str]:
        """Return {'category': emergency|non-emergency, 'topic': ...} for a query."""
        query_lower = query.lower()

        if any(k in query_lower for k in EMERGENCY_KEYWORDS):
            return {"category": "emergency", "topic": "emergency"}

        if re.search(r"\b(no|not|n't)\s+\w+", query_lower) and any(
                k in query_lower for k in MEDICATION_KEYWORDS):
            # Negated medication talk ("I'm not taking my meds") is still medication-scoped.
            return {"category": "non-emergency", "topic": "medication"}

        if any(k in query_lower for k in MEDICATION_KEYWORDS):
            return {"category": "non-emergency", "topic": "medication"}

        if any(k in query_lower for k in MENTAL_HEALTH_KEYWORDS):
            return {"category": "non-emergency", "topic": "mental_health"}

        return {"category": "non-emergency", "topic": "general_health"}
