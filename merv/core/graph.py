"""LangGraph workflow definition — the central Jarvis Core.

Graph structure (Phase 0):

  [input] → [supervisor] → [intent_router] → ...
                                 ↓
              ┌──────────────────┼────────────────┐
              ↓                  ↓                ↓
        [morning_routine]  [family_skill]   [conversation]
              ↓                  ↓                ↓
              └──────────────────┴────────────────┘
                                 ↓
                            [output]
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from merv.config.settings import get_settings
from merv.core.intent_router import classify_intent
from merv.core.personality import get_system_prompt
from merv.core.state import IntentType, MervState

logger = logging.getLogger(__name__)


# ── Node implementations ───────────────────────────────────────────────────────

async def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Entry point: hydrates state, adds greeting context if needed."""
    merv_state = MervState.from_langgraph_dict(state)
    # Supervisor just passes through; routing happens in intent_router_node
    return merv_state.to_langgraph_dict()


async def intent_router_node(state: dict[str, Any]) -> dict[str, Any]:
    """Classifies the user's intent and sets next_node for routing."""
    merv_state = MervState.from_langgraph_dict(state)
    settings = get_settings()

    llm = None
    if settings.anthropic_api_key:
        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        )

    intent, confidence = await classify_intent(merv_state.current_input, llm=llm)
    merv_state.intent = intent
    merv_state.intent_confidence = confidence

    # Map intent → graph node name
    node_map = {
        IntentType.MORNING_ROUTINE: "morning_routine",
        IntentType.FAMILY_ASSISTANT: "family_skill",
        IntentType.PROJECT_MANAGER: "project_skill",
        IntentType.ENGINEERING: "engineering_skill",
        IntentType.CONVERSATION: "conversation",
        IntentType.UNKNOWN: "conversation",
    }
    merv_state.next_node = node_map.get(intent, "conversation")
    logger.info("Intent classified: %s (%.2f) → %s", intent, confidence, merv_state.next_node)
    return merv_state.to_langgraph_dict()


async def morning_routine_node(state: dict[str, Any]) -> dict[str, Any]:
    """Delegates to the MorningRoutineSkill — full brief assembly."""
    from merv.skills.morning_routine import MorningRoutineSkill

    merv_state = MervState.from_langgraph_dict(state)
    skill = MorningRoutineSkill()
    response = await skill.run(merv_state)
    merv_state.add_assistant_message(response, metadata={"skill": "morning_routine"})
    merv_state.routed_to = "morning_routine"
    merv_state.completed = True
    return merv_state.to_langgraph_dict()


async def family_skill_node(state: dict[str, Any]) -> dict[str, Any]:
    """Delegates to FamilyAssistantSkill for calendar/logistics queries."""
    from merv.skills.family_assistant import FamilyAssistantSkill

    merv_state = MervState.from_langgraph_dict(state)
    skill = FamilyAssistantSkill()
    response = await skill.run(merv_state)
    merv_state.add_assistant_message(response, metadata={"skill": "family_assistant"})
    merv_state.routed_to = "family_skill"
    merv_state.completed = True
    return merv_state.to_langgraph_dict()


async def conversation_node(state: dict[str, Any]) -> dict[str, Any]:
    """General-purpose conversational response using Merv's personality."""
    merv_state = MervState.from_langgraph_dict(state)
    settings = get_settings()

    if not settings.anthropic_api_key:
        merv_state.add_assistant_message(
            "I'm Merv, your personal AI companion! I need an Anthropic API key to "
            "have a real conversation. Please set ANTHROPIC_API_KEY in your .env file."
        )
        merv_state.completed = True
        return merv_state.to_langgraph_dict()

    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=merv_state.current_input),
    ]
    response = await llm.ainvoke(messages)
    merv_state.add_assistant_message(response.content)
    merv_state.routed_to = "conversation"
    merv_state.completed = True
    return merv_state.to_langgraph_dict()


async def project_skill_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase 1 placeholder — routes back to conversation with a note."""
    merv_state = MervState.from_langgraph_dict(state)
    merv_state.add_assistant_message(
        "Project management capabilities are coming in Phase 1! For now I've noted "
        "your request. In the meantime, feel free to ask about family scheduling or "
        "your morning brief."
    )
    merv_state.routed_to = "project_skill"
    merv_state.completed = True
    return merv_state.to_langgraph_dict()


async def engineering_skill_node(state: dict[str, Any]) -> dict[str, Any]:
    """Delegates to CodingSkill — ReAct agent that writes, runs, and tests code."""
    from merv.skills.coding_skill import CodingSkill

    merv_state = MervState.from_langgraph_dict(state)
    skill = CodingSkill()
    response = await skill.run(merv_state)
    merv_state.add_assistant_message(response, metadata={"skill": "coding"})
    merv_state.routed_to = "engineering_skill"
    merv_state.completed = True
    return merv_state.to_langgraph_dict()


# ── Conditional routing ────────────────────────────────────────────────────────

def route_by_intent(state: dict[str, Any]) -> str:
    """LangGraph conditional edge: returns the next node name."""
    return state.get("next_node", "conversation")


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_merv_graph() -> "MervGraph":
    """Assemble and compile the Phase 0 LangGraph workflow."""
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("morning_routine", morning_routine_node)
    builder.add_node("family_skill", family_skill_node)
    builder.add_node("project_skill", project_skill_node)
    builder.add_node("engineering_skill", engineering_skill_node)
    builder.add_node("conversation", conversation_node)

    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "morning_routine": "morning_routine",
            "family_skill": "family_skill",
            "project_skill": "project_skill",
            "engineering_skill": "engineering_skill",
            "conversation": "conversation",
        },
    )
    builder.add_edge("morning_routine", END)
    builder.add_edge("family_skill", END)
    builder.add_edge("project_skill", END)
    builder.add_edge("engineering_skill", END)
    builder.add_edge("conversation", END)

    compiled = builder.compile()
    return MervGraph(compiled)


class MervGraph:
    """Thin wrapper around the compiled LangGraph for ergonomic invocation."""

    def __init__(self, graph):
        self._graph = graph

    async def process(self, user_input: str, session_id: str = "", user_id: str | None = None) -> MervState:
        """Process a single user message and return the updated MervState."""
        import uuid
        state = MervState(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
        )
        state.add_user_message(user_input)

        result = await self._graph.ainvoke(state.to_langgraph_dict())
        return MervState.from_langgraph_dict(result)

    async def process_state(self, state: MervState) -> MervState:
        """Process a pre-built state (used by morning routine scheduler)."""
        result = await self._graph.ainvoke(state.to_langgraph_dict())
        return MervState.from_langgraph_dict(result)
