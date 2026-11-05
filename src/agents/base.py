"""Base class and shared types for all MediBot specialist agents."""
import logging
from typing import Optional

from pydantic import Field

from src.llm_manager import get_chat_model

logger = logging.getLogger(__name__)


class AgentResult:
    """Structured result returned by every specialist agent."""

    def __init__(self, agent_name: str, response: str, confidence: float = 1.0,
                 grounded: bool = False, sources: Optional[list] = None, error: Optional[str] = None):
        self.agent_name = agent_name
        self.response = response
        self.confidence = confidence
        self.grounded = grounded
        self.sources = sources or []
        self.error = error

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "response": self.response,
            "confidence": self.confidence,
            "grounded": self.grounded,
            "sources": self.sources,
            "error": self.error,
        }


class BaseAgent:
    """Common functionality: role-bound failover LLM and prompt invocation."""

    name: str = "base"
    role: str = "medical"  # key into ROLE_MODELS / ROLE_LLM_CHAINS

    def __init__(self, **kwargs):
        # Allow tests to inject a temperature/max_tokens override.
        self._model_kwargs = kwargs

    @property
    def llm(self):
        """Failover chat model bound to this agent's role."""
        return get_chat_model(self.role)

    def invoke(self, prompt: str) -> str:
        """Invoke the role's failover LLM with a plain prompt."""
        from langchain_core.messages import HumanMessage

        result = self.llm._generate([HumanMessage(content=prompt)])
        return result.generations[0].message.content
