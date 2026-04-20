"""Unit tests for MervState — the central LangGraph state object."""

import pytest
from datetime import datetime
from merv.core.state import MervState, ConversationTurn, IntentType, FamilyContext


class TestMervState:
    def test_default_state_is_valid(self):
        state = MervState()
        assert state.intent == IntentType.UNKNOWN
        assert state.messages == []
        assert state.completed is False

    def test_add_user_message(self):
        state = MervState()
        state.add_user_message("Hello Merv")
        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hello Merv"
        assert state.current_input == "Hello Merv"

    def test_add_assistant_message(self):
        state = MervState()
        state.add_assistant_message("Good morning!", metadata={"skill": "morning_routine"})
        assert len(state.messages) == 1
        assert state.messages[0].role == "assistant"
        assert state.messages[0].content == "Good morning!"
        assert state.messages[0].metadata == {"skill": "morning_routine"}
        assert state.current_response == "Good morning!"

    def test_round_trip_serialization(self):
        state = MervState(session_id="test-123")
        state.add_user_message("Morning brief please")
        state.intent = IntentType.MORNING_ROUTINE
        state.intent_confidence = 0.9

        data = state.to_langgraph_dict()
        restored = MervState.from_langgraph_dict(data)

        assert restored.session_id == "test-123"
        assert restored.intent == IntentType.MORNING_ROUTINE
        assert restored.intent_confidence == 0.9
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Morning brief please"

    def test_multiple_messages_preserved(self):
        state = MervState()
        state.add_user_message("Hi")
        state.add_assistant_message("Hello!")
        state.add_user_message("What's on today?")

        data = state.to_langgraph_dict()
        restored = MervState.from_langgraph_dict(data)

        assert len(restored.messages) == 3
        assert restored.messages[-1].content == "What's on today?"

    def test_family_context_serializes(self):
        state = MervState()
        state.family_context = FamilyContext(
            member="Henry",
            date="2026-04-20",
            rain_risk=True,
            suggestions=["Reschedule soccer"],
        )
        data = state.to_langgraph_dict()
        restored = MervState.from_langgraph_dict(data)
        assert restored.family_context is not None
        assert restored.family_context.member == "Henry"
        assert restored.family_context.rain_risk is True


class TestConversationTurn:
    def test_default_timestamp(self):
        turn = ConversationTurn(role="user", content="hello")
        assert isinstance(turn.timestamp, datetime)

    def test_valid_roles(self):
        for role in ("user", "assistant", "system"):
            turn = ConversationTurn(role=role, content="test")
            assert turn.role == role
