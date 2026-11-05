"""Mental Health Agent - mood tracking and CBT-style support prompts."""
import logging

from src.agents.base import BaseAgent
from src.prompt import cbt_exercises_prompt, mood_tracking_prompt

logger = logging.getLogger(__name__)


class MentalHealthAgent(BaseAgent):
    """Runs the legacy mood-tracking and CBT prompts on the failover mental-health LLM."""

    name = "mental_health"
    role = "mental_health"

    def mood_tracking(self, description: str, mood_score, tags) -> str:
        prompt = mood_tracking_prompt.format(
            description=description[:200],
            mood_score=mood_score,
            tags=", ".join(tags[:5]),
        )
        return self.invoke(prompt)

    def cbt_exercises(self, concern: str, tried_strategies: str, desired_outcome: str) -> str:
        prompt = cbt_exercises_prompt.format(
            concern=concern[:200],
            tried_strategies=tried_strategies[:200],
            desired_outcome=desired_outcome[:200],
        )
        return self.invoke(prompt)
