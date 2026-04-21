"""Integration tests for LangGraph routing — full workflow without LLM calls.

These tests exercise the graph's routing logic end-to-end using mocked skills,
verifying that:
1. Each intent routes to the correct node
2. State flows correctly through the graph
3. The morning routine triggers the right skill
4. The fallback text path works without ANTHROPIC_API_KEY
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from merv.core.graph import build_merv_graph
from merv.core.state import IntentType, MervState


@pytest.fixture
def graph():
    return build_merv_graph()


class TestGraphIntentRouting:
    """Test that the graph routes to the correct nodes."""

    @pytest.mark.asyncio
    async def test_morning_brief_routes_correctly(self, graph):
        """'Give me my morning brief' → morning_routine node → populated response."""
        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Good morning! Here's your brief..."
            state = await graph.process("Give me my morning brief")

        assert state.intent == IntentType.MORNING_ROUTINE
        assert state.routed_to == "morning_routine"
        assert state.current_response == "Good morning! Here's your brief..."
        assert state.completed is True

    @pytest.mark.asyncio
    async def test_family_query_routes_to_family_skill(self, graph):
        with patch("merv.skills.family_assistant.FamilyAssistantSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Henry has soccer at 4 PM."
            state = await graph.process("What's Henry up to today?")

        assert state.intent == IntentType.FAMILY_ASSISTANT
        assert state.routed_to == "family_skill"
        assert state.current_response == "Henry has soccer at 4 PM."

    @pytest.mark.asyncio
    async def test_project_query_routes_to_project_skill(self, graph):
        state = await graph.process("What tasks are due this week?")
        assert state.intent == IntentType.PROJECT_MANAGER
        assert state.routed_to is not None
        assert "Phase 1" in state.current_response or "project" in state.current_response.lower()

    @pytest.mark.asyncio
    async def test_engineering_query_routes_to_engineering_skill(self, graph):
        state = await graph.process("Fix the bug in the auth module")
        assert state.intent == IntentType.ENGINEERING
        assert state.routed_to is not None

    @pytest.mark.asyncio
    async def test_conversation_returns_response(self, graph):
        """Without API key, conversation node returns a setup message."""
        state = await graph.process("Hello!")
        assert isinstance(state.current_response, str)
        assert len(state.current_response) > 10
        assert state.completed is True

    @pytest.mark.asyncio
    async def test_state_preserves_session_id(self, graph):
        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Brief here."
            state = await graph.process(
                "morning brief",
                session_id="test-session-123",
                user_id="user-456",
            )

        assert state.session_id == "test-session-123"
        assert state.user_id == "user-456"

    @pytest.mark.asyncio
    async def test_messages_recorded_in_state(self, graph):
        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Morning brief content"
            state = await graph.process("Good morning Merv")

        user_messages = [m for m in state.messages if m.role == "user"]
        assistant_messages = [m for m in state.messages if m.role == "assistant"]
        assert len(user_messages) >= 1
        assert len(assistant_messages) >= 1


class TestGraphMorningRoutineWorkflow:
    """Detailed tests of the morning routine workflow."""

    @pytest.mark.asyncio
    async def test_morning_routine_uses_skill(self, graph):
        """Verify the graph actually calls MorningRoutineSkill.run."""
        call_count = 0

        async def mock_run(self_skill, state):
            nonlocal call_count
            call_count += 1
            return "Morning brief assembled."

        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new=mock_run):
            await graph.process("What's my morning brief?")

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_morning_routine_response_in_state(self, graph):
        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "📅 Today's schedule:\n• Henry — Soccer at 4 PM\n— Merv"
            state = await graph.process("Give me my morning brief")

        assert "Soccer" in state.current_response
        assert "Merv" in state.current_response


class TestGraphEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_input_doesnt_crash(self, graph):
        state = await graph.process("")
        assert isinstance(state.current_response, str)

    @pytest.mark.asyncio
    async def test_long_input_handled(self, graph):
        long_input = "What's " + "on my calendar today? " * 50
        state = await graph.process(long_input)
        assert isinstance(state.current_response, str)

    @pytest.mark.asyncio
    async def test_error_in_skill_propagates_gracefully(self, graph):
        async def raise_error(self_skill, state):
            raise RuntimeError("External service down")

        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new=raise_error):
            with pytest.raises(RuntimeError):
                await graph.process("morning brief")
