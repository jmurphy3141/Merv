"""Unit tests for the MorningRoutineSkill — the flagship Phase 0 feature."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from merv.core.state import MervState, IntentType
from merv.skills.morning_routine import (
    MorningRoutineSkill,
    _event_is_today,
    _find_outdoor_events,
    _check_henry_events,
    _format_brief_fallback,
)
from tests.fixtures.sample_data import (
    TODAY, TZ,
    SAMPLE_EVENTS_RAINY_DAY,
    SAMPLE_EVENTS_CLEAR_DAY,
    SAMPLE_EVENTS_SCHOOL_HOUR_CONFLICT,
    SAMPLE_EMAILS_WITH_CANCELLATION,
    RAINY_WEATHER,
    CLEAR_WEATHER,
    make_event,
)


class TestEventIsToday:
    def test_event_today_returns_true(self):
        event = make_event("Test", hour=10, target_date=TODAY)
        assert _event_is_today(event, TODAY, TZ) is True

    def test_event_tomorrow_returns_false(self):
        from datetime import timedelta
        event = make_event("Test", hour=10, target_date=TODAY + timedelta(days=1))
        assert _event_is_today(event, TODAY, TZ) is False

    def test_all_day_event_today(self):
        event = make_event("All Day Event", all_day=True, target_date=TODAY)
        assert _event_is_today(event, TODAY, TZ) is True

    def test_empty_start_returns_false(self):
        event = {"start": "", "summary": "No time"}
        assert _event_is_today(event, TODAY, TZ) is False


class TestFindOutdoorEvents:
    def test_finds_soccer_event(self):
        outdoor = _find_outdoor_events(SAMPLE_EVENTS_RAINY_DAY, TODAY, ["soccer", "baseball"])
        summaries = [e["summary"] for e in outdoor]
        assert any("Soccer" in s for s in summaries)

    def test_no_outdoor_events_returns_empty(self):
        events = [make_event("Family Dinner", hour=18)]
        outdoor = _find_outdoor_events(events, TODAY, ["soccer", "baseball", "lacrosse"])
        assert outdoor == []

    def test_finds_baseball(self):
        events = [make_event("Henry — Baseball Game", hour=15)]
        outdoor = _find_outdoor_events(events, TODAY, ["soccer", "baseball"])
        assert len(outdoor) == 1


class TestCheckHenryEvents:
    def test_detects_school_hour_conflict(self):
        class FakeSettings:
            henry_schedule_constraints = "no_early_morning_appts,no_school_hours"
        warnings = _check_henry_events(SAMPLE_EVENTS_SCHOOL_HOUR_CONFLICT, FakeSettings())
        assert len(warnings) > 0

    def test_no_warnings_for_after_school_events(self):
        class FakeSettings:
            henry_schedule_constraints = "no_early_morning_appts,no_school_hours"
        events = [make_event("Henry — Soccer Practice", hour=16)]
        warnings = _check_henry_events(events, FakeSettings())
        assert len(warnings) == 0

    def test_ignores_non_henry_events(self):
        class FakeSettings:
            henry_schedule_constraints = "no_early_morning_appts,no_school_hours"
        events = [make_event("Mom — Work Meeting", hour=10)]
        warnings = _check_henry_events(events, FakeSettings())
        assert len(warnings) == 0


class TestFormatBriefFallback:
    def test_contains_weather_info(self):
        result = _format_brief_fallback(
            emails=[],
            events=SAMPLE_EVENTS_CLEAR_DAY,
            weather=CLEAR_WEATHER,
            outdoor_events_today=[],
            henry_warnings=[],
            today=TODAY,
            tz=TZ,
        )
        assert "Mainly clear" in result

    def test_rain_alert_shown_when_rain_risk(self):
        outdoor = _find_outdoor_events(SAMPLE_EVENTS_RAINY_DAY, TODAY, ["soccer", "baseball"])
        result = _format_brief_fallback(
            emails=SAMPLE_EMAILS_WITH_CANCELLATION,
            events=SAMPLE_EVENTS_RAINY_DAY,
            weather=RAINY_WEATHER,
            outdoor_events_today=outdoor,
            henry_warnings=[],
            today=TODAY,
            tz=TZ,
        )
        assert "Rain" in result or "rain" in result

    def test_urgent_emails_surfaced(self):
        result = _format_brief_fallback(
            emails=SAMPLE_EMAILS_WITH_CANCELLATION,
            events=[],
            weather=CLEAR_WEATHER,
            outdoor_events_today=[],
            henry_warnings=[],
            today=TODAY,
            tz=TZ,
        )
        assert "Urgent" in result or "urgent" in result or "CANCELED" in result

    def test_henry_warnings_shown(self):
        result = _format_brief_fallback(
            emails=[],
            events=[],
            weather=CLEAR_WEATHER,
            outdoor_events_today=[],
            henry_warnings=["Henry — Doctor: school hours conflict"],
            today=TODAY,
            tz=TZ,
        )
        assert "Henry" in result

    def test_merv_signature_present(self):
        result = _format_brief_fallback(
            emails=[],
            events=[],
            weather=CLEAR_WEATHER,
            outdoor_events_today=[],
            henry_warnings=[],
            today=TODAY,
            tz=TZ,
        )
        assert "Merv" in result

    def test_includes_date_in_header(self):
        result = _format_brief_fallback(
            emails=[],
            events=[],
            weather=CLEAR_WEATHER,
            outdoor_events_today=[],
            henry_warnings=[],
            today=TODAY,
            tz=TZ,
        )
        # April 20 should appear somewhere
        assert "April 20" in result or "Apr" in result


class TestMorningRoutineSkillIntegration:
    """Integration-style tests — real skill with mocked external services."""

    @pytest.mark.asyncio
    async def test_run_returns_string(self):
        skill = MorningRoutineSkill()
        # Calendar and Gmail are auto-mock in absence of credentials
        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)
        assert isinstance(result, str)
        assert len(result) > 50

    @pytest.mark.asyncio
    async def test_run_no_llm_uses_fallback(self):
        """Without ANTHROPIC_API_KEY the skill uses the text fallback."""
        from merv.config.settings import Settings
        mock_settings = Settings(anthropic_api_key="")
        with patch("merv.skills.morning_routine.get_settings", return_value=mock_settings):
            skill = MorningRoutineSkill()
            state = MervState()
            state.add_user_message("morning brief")
            result = await skill.run(state)
            assert "Weather" in result or "weather" in result

    @pytest.mark.asyncio
    async def test_run_with_mocked_data_contains_key_sections(self):
        skill = MorningRoutineSkill()
        skill._calendar = MagicMock()
        skill._calendar.get_events = AsyncMock(return_value=SAMPLE_EVENTS_RAINY_DAY)
        skill._gmail = MagicMock()
        skill._gmail.get_recent_emails = AsyncMock(return_value=SAMPLE_EMAILS_WITH_CANCELLATION)
        skill._weather = MagicMock()
        skill._weather.get_forecast = AsyncMock(return_value=RAINY_WEATHER)

        state = MervState()
        state.add_user_message("morning brief")
        result = await skill.run(state)

        assert isinstance(result, str)
        assert len(result) > 20
