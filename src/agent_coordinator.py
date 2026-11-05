"""Coordinator agent: LangGraph orchestration of the specialist agents.

Flow:
  route -> retrieve -> [specialist agent] -> validate -> END

The router uses deterministic keyword classification (legacy behaviour); the
specialist agent is selected from the routed topic; the validation agent
always checks the draft before it is returned.
"""
import logging
import re
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.agents.base import AgentResult
from src.agents.emergency_agent import EmergencyAgent
from src.agents.general_health_agent import GeneralHealthAgent
from src.agents.medication_agent import MedicationAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.router_agent import RouterAgent
from src.agents.validation_agent import ValidationAgent
from src.llm_manager import LLMAllProvidersFailedError, LLMError
from src.prompt import system_prompt  # noqa: F401  (documented legacy prompt; kept importable)
from src.text_utils import preprocess_text
from video_link import get_first_aid_video
from src.helper_pros import find_relevant_professionals

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Routes queries to specialist agents and validates their responses."""

    def __init__(self, retriever: Optional[RetrievalAgent] = None,
                 max_professionals: int = 1):
        self.router = RouterAgent()
        self.retriever = retriever or RetrievalAgent()
        self.medication_agent = MedicationAgent()
        self.general_health_agent = GeneralHealthAgent()
        self.emergency_agent = EmergencyAgent()
        self.validation_agent = ValidationAgent()
        self.max_professionals = max_professionals

        graph = StateGraph(dict)
        graph.add_node("route", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("specialist", self._specialist_node)
        graph.add_node("validate", self._validate_node)
        graph.set_entry_point("route")
        graph.add_edge("route", "retrieve")
        graph.add_edge("retrieve", "specialist")
        graph.add_edge("specialist", "validate")
        graph.add_edge("validate", END)
        self.graph = graph.compile()

    # ------------------------------------------------------------------
    # Graph nodes (each mutates the shared state dict)
    # ------------------------------------------------------------------
    def _route_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.router.classify(state["query"])
        state["category"] = decision["category"]
        state["topic"] = decision["topic"]
        state["video_link"] = (
            get_first_aid_video(state["query"]) if decision["category"] == "emergency" else None
        )
        return state

    def _retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            docs = self.retriever.retrieve(state["query"], k=2)
        except Exception as exc:  # config-level retrieval error -> degrade to no context
            logger.error("Retrieval failed, continuing without context: %s", exc)
            docs = []
        state["docs"] = docs
        state["context"] = "\n".join(d.page_content for d in docs)
        return state

    def _specialist_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        topic = state.get("topic", "general_health")
        query, context, history = state["query"], state.get("context", ""), state.get("history", "")
        try:
            if topic == "emergency":
                response = self.emergency_agent.run(query, context, history)
            elif topic == "medication":
                response = self.medication_agent.run(query, context, history)
            else:
                response = self.general_health_agent.run(query, context, history)
            state["draft"] = response
            state["agent_used"] = (
                "emergency" if topic == "emergency"
                else "medication" if topic == "medication"
                else "general_health"
            )
        except LLMAllProvidersFailedError as exc:
            logger.error("All providers failed for topic '%s': %s", topic, exc)
            state["draft"] = ""
            state["agent_used"] = topic
            state["agent_error"] = "All LLM providers are currently unavailable. Please try again shortly."
        except LLMError as exc:
            logger.error("LLM error for topic '%s': %s", topic, exc)
            state["draft"] = ""
            state["agent_used"] = topic
            state["agent_error"] = "The assistant is temporarily unavailable. Please try again shortly."
        return state

    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        draft = state.get("draft", "")
        context = state.get("context", "")
        query = state["query"]

        if state.get("agent_error") and not draft:
            state["response"] = state["agent_error"]
            state["validation"] = "provider_failure"
            return state

        result = self.validation_agent.validate(draft, context, query)
        if result["passed"]:
            state["response"] = result["response"]
            state["validation"] = "passed"
        else:
            # One regeneration attempt with a stricter grounding instruction.
            logger.warning("Validation failed (%s) for query: %.80s", result["reason"], query)
            stricter = (
                "IMPORTANT: Your previous answer was rejected because it contained claims not "
                "supported by the retrieved context. Answer again using ONLY the context below. "
                "If the context is insufficient, say exactly what information is missing and "
                "advise consulting a healthcare professional. Do not invent doses or diagnoses.\n\n"
                f"Retrieved context:\n{context or '(none retrieved)'}\n\n"
                f"User question: {query}\n\nAnswer:"
            )
            try:
                retry_draft = self.general_health_agent.invoke(stricter)
                retry_result = self.validation_agent.validate(retry_draft, context, query)
            except LLMError as exc:
                logger.error("Regeneration attempt failed: %s", exc)
                retry_result = {"passed": False, "response": "", "reason": "regeneration_failed"}

            if retry_result.get("passed"):
                state["response"] = retry_result["response"]
                state["validation"] = "passed_after_retry"
            else:
                state["response"] = (
                    "I couldn't generate a sufficiently reliable answer to this question. "
                    "Please try rephrasing your question or contact a healthcare professional."
                )
                state["validation"] = f"failed:{result['reason']}"
        return state

    # ------------------------------------------------------------------
    # Enrichments and public entry point
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_response(response: str) -> str:
        response = re.sub(r"(as an ai(?: language)? model|i(?:'m| am) an? ai)", "",
                          response, flags=re.IGNORECASE)
        return re.sub(r"[ \t]+", " ", response).strip()

    def _append_professionals(self, response: str, query: str) -> str:
        try:
            pros = find_relevant_professionals(query, max_results=self.max_professionals)
        except Exception as exc:
            logger.warning("Professional lookup failed: %s", exc)
            pros = []
        if pros:
            response += "\n\nFor specialized care at Metro Hospital, you may contact:"
            for p in pros:
                response += f"\n- {p['name']} ({p['specialty']}), 📞 {p.get('phone', 'N/A')}"
        return response

    def handle(self, query: str, history=None) -> dict:
        """Process a chat query end-to-end. Returns the legacy API payload dict."""
        # Normalize history the same way the legacy endpoint did.
        history = history or []
        if history and isinstance(history[0], dict):
            history = [f"{m.get('sender', 'unknown')}: {m.get('text', '')}" for m in history]
        history_text = "\n".join(history[-2:]) if history else ""

        state: Dict[str, Any] = {
            "query": query,
            "history": history_text,
            "query_terms": preprocess_text(query),
        }
        final = self.graph.invoke(state)

        response = self._clean_response(final.get("response", ""))
        category = final.get("category", "non-emergency")
        video_link = final.get("video_link")

        response = self._append_professionals(response, query)
        if video_link:
            response += f"\n\n**First‑Aid Video:** {video_link}"

        return {
            "response": response,
            "category": category,
            "video_link": video_link,
            "agent": final.get("agent_used"),
            "validation": final.get("validation"),
        }
