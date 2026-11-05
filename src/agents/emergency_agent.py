"""Emergency Agent - first-aid guidance using the emergency protocol prompt."""
import logging

from src.agents.base import BaseAgent
from src.prompt import emergency_subprompt

logger = logging.getLogger(__name__)

EMERGENCY_PROMPT = (
    "You are MediBot's Emergency Agent for Metro Hospital (Nairobi). The user's message was "
    "flagged as a possible EMERGENCY. Respond with the emergency protocol structure shown "
    "below, adapted to the user's situation. Be direct and actionable.\n\n"
    "Retrieved first-aid context (use if relevant):\n{context}\n\n"
    "Conversation history:\n{history}\n\n"
    "User message: {query}\n\n"
    "{protocol}\n\n"
    "Respond now with the emergency response for this situation."
)


class EmergencyAgent(BaseAgent):
    name = "emergency"
    role = "medical"

    def run(self, query: str, context: str, history: str = "") -> str:
        prompt = EMERGENCY_PROMPT.format(
            context=context or "(none retrieved)", history=history, query=query, protocol=emergency_subprompt
        )
        return self.invoke(prompt)
