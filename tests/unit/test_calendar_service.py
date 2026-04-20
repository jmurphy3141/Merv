"""Unit tests for Google Calendar service and Henry constraint checking."""

import pytest
from datetime import date
from zoneinfo import ZoneInfo

from merv.skills.calendar_service import GoogleCalendarService, check_henry_constraints
from tests.fixtures.sample_data import TODAY, make_event


class TestGoogleCalendarServiceMock:
    """Tests for mock calendar mode (no credentials needed)."""

    def setup_method(self):
        self.service = GoogleCalendarService(mock=True)

    @pytest.mark.asyncio
    async def test_returns_events_for_today(self):
        events = await self.service.get_events(
            calendar_ids=["primary"],
            target_date=TODAY,
            timezone="America/New_York",
        )
        assert isinstance(events, list)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_events_have_required_fields(self):
        events = await self.service.get_events(
            calendar_ids=["primary"],
            target_date=TODAY,
        )
        required_fields = {"id", "calendar_id", "summary", "start", "end", "status"}
        for event in events:
            assert required_fields.issubset(event.keys()), f"Missing fields in event: {event}"

    @pytest.mark.asyncio
    async def test_mock_returns_henry_soccer(self):
        events = await self.service.get_events(["primary"], target_date=TODAY)
        summaries = [e["summary"] for e in events]
        assert any("Soccer" in s for s in summaries)

    @pytest.mark.asyncio
    async def test_mock_returns_tentative_appointment(self):
        events = await self.service.get_events(["primary"], target_date=TODAY)
        tentative = [e for e in events if e["status"] == "tentative"]
        assert len(tentative) > 0


class TestCheckHenryConstraints:
    """Tests for constraint violation detection."""

    def test_early_morning_appointment_flagged(self):
        event = make_event("Henry — Doctor", hour=7, minute=30)  # 7:30 AM
        violations = check_henry_constraints(event)
        assert len(violations) > 0
        assert any("early morning" in v.lower() for v in violations)

    def test_school_hour_appointment_flagged(self):
        event = make_event("Henry — Checkup", hour=10, minute=0)  # 10 AM Monday
        violations = check_henry_constraints(event)
        assert len(violations) > 0
        assert any("school hours" in v.lower() for v in violations)

    def test_after_school_appointment_ok(self):
        event = make_event("Henry — Dentist", hour=15, minute=30)  # 3:30 PM Monday
        violations = check_henry_constraints(event)
        assert len(violations) == 0

    def test_weekend_school_hour_not_flagged(self):
        saturday = date(2026, 4, 25)  # Saturday
        event = make_event("Henry — Soccer Tournament", hour=10, target_date=saturday)
        violations = check_henry_constraints(event)
        # Weekend shouldn't trigger school-hours constraint
        assert not any("school hours" in v.lower() for v in violations)

    def test_all_day_event_no_violations(self):
        event = make_event("Henry — School Trip", hour=0, all_day=True)
        violations = check_henry_constraints(event)
        assert violations == []

    def test_returns_list(self):
        event = make_event("Henry — Appointment", hour=14)
        result = check_henry_constraints(event)
        assert isinstance(result, list)
