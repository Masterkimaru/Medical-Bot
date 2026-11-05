"""Medication Information Agent - answers medication questions from retrieved context."""
import logging

from src.agents.base import BaseAgent
from src.prompt import system_prompt  # noqa: F401  (kept for prompt-consistency reference)

logger = logging.getLogger(__name__)

MEDICATION_PROMPT = (
    "You are MediBot's Medication Information Agent for Metro Hospital (Nairobi).\n\n"
    "Answer the user's question about a medication using ONLY the retrieved reference "
    "information below. If the reference information does not contain the answer, say so "
    "plainly and recommend speaking to a pharmacist or doctor instead of guessing.\n\n"
    "Rules:\n"
    "- Always attach 'as prescribed' to any dosage mention.\n"
    "- Mention common side effects and key interactions when the reference includes them.\n"
    "- Never invent drug names, doses or interactions.\n"
    "- Preserve negations exactly ('no allergies' must not become 'allergies').\n"
    "- End with: 'Disclaimer: Always consult a healthcare professional about medications.'\n\n"
    "Retrieved reference information:\n{context}\n\n"
    "Conversation history:\n{history}\n\n"
    "User question: {query}\n\n"
    "Answer:"
)


class MedicationAgent(BaseAgent):
    name = "medication"
    role = "medical"

    def run(self, query: str, context: str, history: str = "") -> str:
        prompt = MEDICATION_PROMPT.format(context=context or "(none retrieved)", history=history, query=query)
        return self.invoke(prompt)
