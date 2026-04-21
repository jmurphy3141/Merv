"""Morning Routine Skill — the flagship Phase 0 feature.

Assembles the full daily brief:
  1. Email roundup (urgent + family-related first)
  2. Today's schedule across all family calendars
  3. Weather check with rain-out analysis for outdoor sports
  4. Henry's constraint violations flagged proactively
  5. Intelligent suggestions (reschedule, bring umbrella, etc.)

Designed to be sent via Telegram each morning or triggered on demand.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from merv.config.settings import get_settings
from merv.core.personality import get_system_prompt
from merv.skills.base import BaseSkill
from merv.skills.calendar_service import GoogleCalendarService, check_henry_constraints
from merv.skills.email_service import GmailService
from merv.skills.weather_service import WeatherService

if TYPE_CHECKING:
    from merv.core.state import MervState

logger = logging.getLogger(__name__)


class MorningRoutineSkill(BaseSkill):
    """
    Builds the daily morning brief: emails + schedule + weather + suggestions.
    """

    name = "morning_routine"
    description = "Full morning brief: email roundup, schedule, weather, suggestions"

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._calendar = GoogleCalendarService(
            credentials_file=settings.google_credentials_file,
            token_file=settings.google_token_file,
        )
        self._gmail = GmailService(
            credentials_file=settings.google_credentials_file,
            token_file=settings.google_token_file,
        )
        self._weather = WeatherService()
        self._settings = settings

    async def run(self, state: "MervState") -> str:
        """Assemble and return the morning brief as a formatted string."""
        settings = self._settings
        tz = ZoneInfo(settings.family_timezone)
        today = datetime.now(tz).date()

        self.logger.info("Assembling morning brief for %s", today)

        # Fetch all data sources
        emails = await self._gmail.get_recent_emails(max_results=10, hours_back=16)
        events = await self._calendar.get_events(
            calendar_ids=settings.calendar_ids,
            target_date=today,
            days_ahead=2,  # today + tomorrow for forward planning
            timezone=settings.family_timezone,
        )
        weather = await self._weather.get_forecast(today)
        tomorrow_weather = await self._weather.get_forecast(
            today + timedelta(days=1)
        )

        # Determine rain-out risks
        outdoor_events_today = _find_outdoor_events(events, today, settings.rain_out_sports_list)
        outdoor_events_tomorrow = _find_outdoor_events(
            events, today + timedelta(days=1), settings.rain_out_sports_list
        )

        # Constraint checks for Henry
        henry_warnings = _check_henry_events(events, settings)

        # Build the brief
        if settings.anthropic_api_key:
            return await self._llm_brief(
                emails=emails,
                events=[e for e in events if _event_is_today(e, today, tz)],
                weather=weather,
                tomorrow_weather=tomorrow_weather,
                outdoor_events_today=outdoor_events_today,
                outdoor_events_tomorrow=outdoor_events_tomorrow,
                henry_warnings=henry_warnings,
                today=today,
                tz=tz,
            )

        return _format_brief_fallback(
            emails=emails,
            events=[e for e in events if _event_is_today(e, today, tz)],
            weather=weather,
            outdoor_events_today=outdoor_events_today,
            henry_warnings=henry_warnings,
            today=today,
            tz=tz,
        )

    async def _llm_brief(self, **kwargs) -> str:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = self._settings
        context = _build_brief_context(**kwargs)
        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.4,
            max_tokens=2048,
        )
        messages = [
            SystemMessage(content=get_system_prompt(extra_context=context)),
            HumanMessage(
                content="Please give me my complete morning brief. "
                "Be concise but thorough — prioritize anything urgent."
            ),
        ]
        response = await llm.ainvoke(messages)
        return response.content


# ── Helper functions ───────────────────────────────────────────────────────────

def _event_is_today(event: dict, today, tz) -> bool:
    start = event.get("start", "")
    if not start:
        return False
    try:
        if "T" in start:
            dt = datetime.fromisoformat(start)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt.date() == today
        else:
            from datetime import date
            return date.fromisoformat(start) == today
    except ValueError:
        return False


def _find_outdoor_events(events: list, target_date, sport_keywords: list[str]) -> list[dict]:
    """Find events that match rain-out sport keywords on a given date."""
    outdoor = []
    for ev in events:
        summary_lower = ev.get("summary", "").lower()
        desc_lower = ev.get("description", "").lower()
        combined = summary_lower + " " + desc_lower
        if any(kw.lower() in combined for kw in sport_keywords):
            outdoor.append(ev)
    return outdoor


def _check_henry_events(events: list, settings) -> list[str]:
    """Check all Henry-related events for constraint violations."""
    warnings = []
    for ev in events:
        summary_lower = ev.get("summary", "").lower()
        if any(kw in summary_lower for kw in ["henry", "pediatrician", "appointment", "dr.", "doctor"]):
            violations = check_henry_constraints(ev)
            for v in violations:
                warnings.append(f"{ev['summary']}: {v}")
    return warnings


def _build_brief_context(
    emails, events, weather, tomorrow_weather,
    outdoor_events_today, outdoor_events_tomorrow,
    henry_warnings, today, tz
) -> str:
    today_str = today.strftime("%A, %B %d, %Y")
    lines = [f"=== MORNING BRIEF DATA for {today_str} ===\n"]

    # Weather
    lines.append("WEATHER TODAY:")
    lines.append(f"  {weather['description']}, High: {weather.get('temp_max', '?')}°F, Low: {weather.get('temp_min', '?')}°F")
    if weather["rain_risk"]:
        lines.append("  ⚠️ RAIN RISK — outdoor activities may be affected")

    lines.append("\nWEATHER TOMORROW:")
    lines.append(f"  {tomorrow_weather['description']}, High: {tomorrow_weather.get('temp_max', '?')}°F")
    if tomorrow_weather["rain_risk"]:
        lines.append("  ⚠️ RAIN RISK tomorrow too")

    # Emails
    lines.append(f"\nEMAILS TO REVIEW ({len(emails)} new):")
    urgent = [e for e in emails if e["urgent"]]
    family = [e for e in emails if e["family_related"] and not e["urgent"]]
    other = [e for e in emails if not e["urgent"] and not e["family_related"]]

    for e in urgent:
        lines.append(f"  🔴 URGENT: {e['subject']} — from {e['sender']}")
        lines.append(f"     {e['snippet'][:120]}")
    for e in family:
        lines.append(f"  👨‍👩‍👦 FAMILY: {e['subject']} — from {e['sender']}")
        lines.append(f"     {e['snippet'][:100]}")
    for e in other[:3]:
        lines.append(f"  📧 {e['subject']}")

    # Today's schedule
    lines.append(f"\nTODAY'S SCHEDULE ({len(events)} events):")
    if not events:
        lines.append("  Clear day — no events scheduled!")
    for ev in events:
        time_str = ""
        if not ev.get("all_day") and "T" in ev.get("start", ""):
            try:
                dt = datetime.fromisoformat(ev["start"])
                time_str = f" @ {dt.strftime('%I:%M %p')}"
            except ValueError:
                pass
        status = f" [{ev['status'].upper()}]" if ev["status"] != "confirmed" else ""
        lines.append(f"  • {ev['summary']}{time_str}{status}")

    # Rain-out alerts
    if outdoor_events_today and weather["rain_risk"]:
        lines.append("\n⚠️ RAIN-OUT RISK TODAY:")
        for ev in outdoor_events_today:
            lines.append(f"  • {ev['summary']} may be canceled due to rain")

    if outdoor_events_tomorrow and tomorrow_weather["rain_risk"]:
        lines.append("\n⚠️ RAIN-OUT RISK TOMORROW:")
        for ev in outdoor_events_tomorrow:
            lines.append(f"  • {ev['summary']} tomorrow also at risk")

    # Henry warnings
    if henry_warnings:
        lines.append("\n⚠️ HENRY SCHEDULING CONCERNS:")
        for w in henry_warnings:
            lines.append(f"  • {w}")

    lines.append("\n=== END OF BRIEF DATA ===")
    lines.append("\nPlease format this as a clear, friendly morning brief. Lead with urgent items.")
    return "\n".join(lines)


def _format_brief_fallback(emails, events, weather, outdoor_events_today, henry_warnings, today, tz) -> str:
    """Plain-text brief when no LLM is configured."""
    today_str = today.strftime("%A, %B %d")
    sections = [f"☀️ Good morning! Here's your brief for {today_str}.\n"]

    # Weather
    sections.append(f"🌤️ **Weather:** {weather['description']}")
    if weather.get("temp_max"):
        sections.append(f"   High: {weather['temp_max']}°F  Low: {weather.get('temp_min', '?')}°F")
    if weather["rain_risk"]:
        sections.append("⚠️ Rain expected today — outdoor activities at risk!\n")

    # Rain-outs
    if outdoor_events_today and weather["rain_risk"]:
        sections.append("⚠️ **Rain-out alerts:**")
        for ev in outdoor_events_today:
            sections.append(f"   • {ev['summary']} — may be canceled")
        sections.append("")

    # Henry warnings
    if henry_warnings:
        sections.append("⚠️ **Henry scheduling concerns:**")
        for w in henry_warnings:
            sections.append(f"   • {w}")
        sections.append("")

    # Emails
    urgent_emails = [e for e in emails if e["urgent"]]
    if urgent_emails:
        sections.append("🔴 **Urgent emails:**")
        for e in urgent_emails:
            sections.append(f"   • {e['subject']}")
        sections.append("")

    family_emails = [e for e in emails if e["family_related"] and not e["urgent"]]
    if family_emails:
        sections.append("👨‍👩‍👦 **Family emails:**")
        for e in family_emails:
            sections.append(f"   • {e['subject']}")
        sections.append("")

    # Schedule
    sections.append(f"📅 **Today's schedule ({len(events)} events):**")
    if not events:
        sections.append("   No events — clear day!")
    for ev in events:
        time_str = ""
        if not ev.get("all_day") and "T" in ev.get("start", ""):
            try:
                dt = datetime.fromisoformat(ev["start"])
                time_str = f" @ {dt.strftime('%I:%M %p')}"
            except ValueError:
                pass
        status_flag = " [TENTATIVE]" if ev.get("status") == "tentative" else ""
        sections.append(f"   • {ev['summary']}{time_str}{status_flag}")

    sections.append("\n— Merv 🤖")
    return "\n".join(sections)
