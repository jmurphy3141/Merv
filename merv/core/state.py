"""LangGraph state definitions for Merv.

All nodes in the graph communicate through MervState — the single shared
object that flows through every workflow. Typed with TypedDict so LangGraph
can merge partial updates from parallel branches.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class IntentType(str, enum.Enum):
    """High-level routing buckets for incoming requests."""
    FAMILY_ASSISTANT = "family_assistant"
    MORNING_ROUTINE = "morning_routine"
    PROJECT_MANAGER = "project_manager"
    ENGINEERING = "engineering"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class ConversationTurn(BaseModel):
    """Single turn in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FamilyContext(BaseModel):
    """Snapshot of family state relevant to the current request."""
    member: Optional[str] = None
    date: Optional[str] = None          # ISO date string
    events: list[dict[str, Any]] = Field(default_factory=list)
    emails: list[dict[str, Any]] = Field(default_factory=list)
    weather: Optional[dict[str, Any]] = None
    rain_risk: Optional[bool] = None
    suggestions: list[str] = Field(default_factory=list)
    constraints_violated: list[str] = Field(default_factory=list)


class MervState(BaseModel):
    """
    Central state object that flows through the LangGraph workflow.

    LangGraph uses TypedDict for raw state, but we wrap in Pydantic for
    validation. The graph converts between the two at boundaries.
    """
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: list[ConversationTurn] = Field(default_factory=list)
    current_input: str = ""
    current_response: str = ""

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: IntentType = IntentType.UNKNOWN
    intent_confidence: float = 0.0
    routed_to: Optional[str] = None     # which skill/sub-agent handled this

    # ── Family Context ────────────────────────────────────────────────────────
    family_context: Optional[FamilyContext] = None

    # ── Workflow control ──────────────────────────────────────────────────────
    next_node: Optional[str] = None     # used by supervisor to direct flow
    error: Optional[str] = None
    completed: bool = False

    # ── Metadata ──────────────────────────────────────────────────────────────
    session_id: str = ""
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_user_message(self, content: str) -> "MervState":
        self.messages.append(ConversationTurn(role="user", content=content))
        self.current_input = content
        self.updated_at = datetime.utcnow()
        return self

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> "MervState":
        self.messages.append(ConversationTurn(
            role="assistant",
            content=content,
            metadata=metadata or {},
        ))
        self.current_response = content
        self.updated_at = datetime.utcnow()
        return self

    def to_langgraph_dict(self) -> dict[str, Any]:
        """Serialize for LangGraph state checkpointing."""
        return self.model_dump(mode="json")

    @classmethod
    def from_langgraph_dict(cls, data: dict[str, Any]) -> "MervState":
        return cls.model_validate(data)
