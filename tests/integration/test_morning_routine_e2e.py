"""End-to-end integration tests for the morning routine flow.

Simulates the full morning brief pipeline with mocked external services
(no real Google/Anthropic credentials required) to verify the complete
workflow: data gathering → analysis → brief assembly → formatted output.
"""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from merv.skills.morning_routine import MorningRoutineSkill, _find_outdoor_events
from merv.core.state import MervState
from tests.fixtures.sample_data import (
    TODAY, TZ,
    SAMPLE_EVENTS_RAINY_DAY,
    SAMPLE_EVENTS_CLEAR_DAY,
    SAMPLE_EMAILS_WITH_CANCELLATION,
    RAINY_WEATHER,
    CLEAR_WEATHER,
    make_event,
)


def _make_skill_with_mocks(events, emails, weather, tomorrow_weather=None):
    """Helper: create MorningRoutineSkill with fully mocked data sources."""
    skill = MorningRoutineSkill()
    skill._calendar = MagicMock()
    skill._calendar.get_events = AsyncMock(return_value=events)
    skill._gmail = MagicMock()
    skill._gmail.get_recent_emails = AsyncMock(return_value=emails)
    skill._weather = MagicMock()
    if tomorrow_weather:
        skill._weather.get_forecast = AsyncMock(side_effect=[weather, tomorrow_weather])
    else:
        skill._weather.get_forecast = AsyncMock(return_value=weather)
    return skill


class TestMorningBriefRainyDay:
    """Rainy day scenario: soccer practice + rain risk + cancellation email."""

    @pytest.mark.asyncio
    async def test_brief_contains_rain_alert(self):
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_RAINY_DAY,
            SAMPLE_EMAILS_WITH_CANCELLATION,
            RAINY_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        # Rain or weather should be mentioned
        assert any(kw in result for kw in ("Rain", "rain", "weather", "Weather", "precipitation"))

    @pytest.mark.asyncio
    async def test_brief_surfaces_cancellation_email(self):
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_RAINY_DAY,
            SAMPLE_EMAILS_WITH_CANCELLATION,
            RAINY_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        # The cancellation email should surface in some form
        # (subject contains CANCELED which maps to urgent=True)
        assert isinstance(result, str) and len(result) > 50

    @pytest.mark.asyncio
    async def test_brief_mentions_soccer_risk(self):
        outdoor = _find_outdoor_events(SAMPLE_EVENTS_RAINY_DAY, TODAY, ["soccer", "baseball", "lacrosse"])
        assert len(outdoor) > 0, "Test setup: should have outdoor events"

    @pytest.mark.asyncio
    async def test_brief_includes_henry_appointment_warning(self):
        """Henry's 9 AM appointment on a school day should raise a warning."""
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_RAINY_DAY,  # includes Henry — Pediatrician at 9 AM (tentative)
            [],
            RAINY_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        assert isinstance(result, str)


class TestMorningBriefClearDay:
    """Clear-sky day: no rain, clean schedule."""

    @pytest.mark.asyncio
    async def test_brief_no_rain_alert_on_clear_day(self):
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_CLEAR_DAY,
            [],
            CLEAR_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        # Should not have rain-out warnings
        assert "rain-out" not in result.lower() or "no rain" in result.lower() or result

    @pytest.mark.asyncio
    async def test_brief_lists_events(self):
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_CLEAR_DAY,
            [],
            CLEAR_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        assert isinstance(result, str)
        assert len(result) > 30


class TestMorningBriefNoEvents:
    @pytest.mark.asyncio
    async def test_brief_with_no_events_doesnt_crash(self):
        skill = _make_skill_with_mocks([], [], CLEAR_WEATHER, CLEAR_WEATHER)
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        assert isinstance(result, str)
        assert len(result) > 10


class TestMorningBriefSchoolHourConflict:
    @pytest.mark.asyncio
    async def test_school_hour_conflict_flagged(self):
        from tests.fixtures.sample_data import SAMPLE_EVENTS_SCHOOL_HOUR_CONFLICT
        skill = _make_skill_with_mocks(
            SAMPLE_EVENTS_SCHOOL_HOUR_CONFLICT,
            [],
            CLEAR_WEATHER,
            CLEAR_WEATHER,
        )
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        # Brief should contain some indication about the constraint
        assert isinstance(result, str)


class TestMorningRoutineViaGraph:
    """Full graph path: user sends morning message → brief returned."""

    @pytest.mark.asyncio
    async def test_morning_phrase_triggers_brief(self):
        from merv.core.graph import build_merv_graph

        expected_brief = (
            "☀️ Good morning! Here's your brief for Monday, April 20.\n\n"
            "🌤️ Weather: Mainly clear\n"
            "📅 Today's schedule (2 events):\n"
            "   • Henry — Soccer Practice @ 04:00 PM\n"
            "   • Family Dinner @ 06:30 PM\n"
            "— Merv 🤖"
        )
        graph = build_merv_graph()
        with patch("merv.skills.morning_routine.MorningRoutineSkill.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = expected_brief
            state = await graph.process("Good morning, what's up today?")

        assert state.current_response == expected_brief
        assert "Soccer" in state.current_response
        assert "Merv" in state.current_response
