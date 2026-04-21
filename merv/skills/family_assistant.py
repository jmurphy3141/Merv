"""Family Assistant Skill — handles ad-hoc family logistics queries.

Answers questions about schedules, constraints, weather impacts, and
provides constraint-aware suggestions for Henry's appointments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from merv.config.settings import get_settings
from merv.core.personality import get_system_prompt
from merv.skills.base import BaseSkill
from merv.skills.calendar_service import GoogleCalendarService, check_henry_constraints
from merv.skills.weather_service import WeatherService

if TYPE_CHECKING:
    from merv.core.state import MervState

logger = logging.getLogger(__name__)


class FamilyAssistantSkill(BaseSkill):
    """
    Handles family logistics: schedules, appointment constraints,
    weather impacts on outdoor activities, and suggestions.
    """

    name = "family_assistant"
    description = "Family scheduling, calendar queries, Henry's constraints, weather impacts"

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._calendar = GoogleCalendarService(
            credentials_file=settings.google_credentials_file,
            token_file=settings.google_token_file,
        )
        self._weather = WeatherService()
        self._settings = settings

    async def run(self, state: "MervState") -> str:
        settings = self._settings
        tz = ZoneInfo(settings.family_timezone)
        today = datetime.now(tz).date()

        # Fetch calendar + weather in parallel context
        events = await self._calendar.get_events(
            calendar_ids=settings.calendar_ids,
            target_date=today,
            timezone=settings.family_timezone,
        )
        weather = await self._weather.get_forecast(today)

        # Build context for LLM response
        context = _build_family_context(events, weather, settings)

        # Use LLM to generate a natural-language answer to the specific question
        if settings.anthropic_api_key:
            return await self._llm_response(state.current_input, context)

        # Fallback: structured text response
        return _format_fallback(events, weather, state.current_input)

    async def _llm_response(self, user_input: str, context: str) -> str:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = self._settings
        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=1024,
        )
        messages = [
            SystemMessage(content=get_system_prompt(extra_context=context)),
            HumanMessage(content=user_input),
        ]
        response = await llm.ainvoke(messages)
        return response.content


def _build_family_context(events: list, weather: dict, settings) -> str:
    tz = ZoneInfo(settings.family_timezone)
    today = datetime.now(tz).strftime("%A, %B %d")

    lines = [f"Today is {today}.", f"Weather: {weather['description']}"]
    if weather["rain_risk"]:
        lines.append("⚠️ RAIN RISK TODAY — outdoor sports may be affected.")

    lines.append("\nToday's calendar events:")
    if not events:
        lines.append("  No events found.")
    for ev in events:
        time_str = ""
        if not ev.get("all_day") and "T" in ev.get("start", ""):
            try:
                dt = datetime.fromisoformat(ev["start"])
                time_str = dt.strftime(" at %I:%M %p")
            except ValueError:
                pass
        status = f" [{ev['status']}]" if ev["status"] != "confirmed" else ""
        lines.append(f"  • {ev['summary']}{time_str}{status}")

        # Henry constraint checks
        if any(kw in ev["summary"].lower() for kw in ["henry", "pediatrician", "appointment", "dr.", "doctor"]):
            violations = check_henry_constraints(ev)
            for v in violations:
                lines.append(f"    {v}")

    lines.append(f"\nRain-out sports to watch: {', '.join(settings.rain_out_sports_list)}")
    lines.append(f"Henry's constraints: {settings.henry_schedule_constraints}")
    return "\n".join(lines)


def _format_fallback(events: list, weather: dict, user_input: str) -> str:
    lines = [
        "Here's what I see for the family today:\n",
        f"**Weather:** {weather['description']}",
    ]
    if weather["rain_risk"]:
        lines.append("⚠️ Rain risk — check outdoor activities!\n")
    if events:
        lines.append("**Events:**")
        for ev in events:
            lines.append(f"  • {ev['summary']}")
    else:
        lines.append("No events scheduled today.")
    return "\n".join(lines)
