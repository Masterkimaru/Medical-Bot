"""Response Validation Agent.

Two layers:
1. Deterministic safety checks (always run, no LLM cost): non-empty output,
   safety disclaimer present, no leaked internal reasoning, no API keys.
2. Optional LLM-grounding check (when VALIDATION_ENABLED=true): verifies the
   draft is supported by the retrieved context; flags unsupported medical
   claims for regeneration with a stricter grounding instruction.
"""
import logging
import re

from src.agents.base import BaseAgent
from src.config import VALIDATION_ENABLED

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = (
    "You are MediBot's Response Validation Agent. Compare the DRAFT answer against the "
    "RETRIEVED CONTEXT. Reply with exactly one line:\n"
    "PASS - if the draft is consistent with the context or is clearly qualified general guidance\n"
    "FAIL - if the draft makes specific medical claims (doses, diagnoses, drug interactions) "
    "that are absent from or contradicted by the context\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "DRAFT:\n{draft}\n\n"
    "Verdict:"
)

_THINK_RE = re.compile(r"◁think▷.*?◁/think▷", re.DOTALL)
_KEY_RE = re.compile(r"(sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9_\-\.]{16,})")


class ValidationAgent(BaseAgent):
    name = "validation"
    role = "validation"

    def validate(self, draft: str, context: str, query: str) -> dict:
        """Return {'passed': bool, 'response': str, 'reason': str}."""
        issues = []

        if not draft or not draft.strip():
            return {"passed": False, "response": "", "reason": "empty_response"}

        cleaned = _THINK_RE.sub("", draft).strip()
        if cleaned != draft.strip():
            issues.append("removed_internal_reasoning")

        if _KEY_RE.search(cleaned):
            logger.error("Validation blocked a response containing what looked like an API key.")
            return {"passed": False, "response": "", "reason": "leaked_credential"}

        if not cleaned.lower().startswith("i couldn't"):
            if "consult a healthcare professional" not in cleaned.lower() and \
               "consult a verified health practitioner" not in cleaned.lower():
                cleaned += "\n\n⚠️ Note: Consult a healthcare professional for medical advice"
                issues.append("added_disclaimer")

        if VALIDATION_ENABLED and context:
            verdict = ""
            try:
                verdict = self.invoke(_GROUNDING_PROMPT.format(context=context[:4000], draft=cleaned[:4000]))
            except Exception as exc:
                logger.warning("Grounding validation LLM failed (%s); accepting rule-based result.", exc)
            if verdict.strip().upper().startswith("FAIL"):
                return {"passed": False, "response": cleaned, "reason": "unsupported_medical_claims"}

        return {"passed": True, "response": cleaned, "reason": ",".join(issues) or "ok"}
