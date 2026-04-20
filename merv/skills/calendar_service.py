"""Google Calendar integration — async wrapper around the Google Calendar API.

Handles OAuth2 credential management, event fetching, and constraint checking.
In development/test mode (no credentials) it returns realistic mock data.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class GoogleCalendarService:
    """
    Fetches events from one or more Google Calendars.

    Pass mock=True (or leave credentials unconfigured) to get realistic
    mock data — useful for demos and tests without live Google credentials.
    """

    def __init__(self, credentials_file: str = "", token_file: str = "", mock: bool = False):
        self._credentials_file = Path(credentials_file).expanduser() if credentials_file else None
        self._token_file = Path(token_file).expanduser() if token_file else None
        self._mock = mock or not self._has_credentials()
        self._service = None

    def _has_credentials(self) -> bool:
        return (
            self._credentials_file is not None
            and self._credentials_file.exists()
        )

    def _get_service(self):
        """Lazy-init the Google Calendar API service."""
        if self._service:
            return self._service
        if self._mock:
            return None

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = [
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/gmail.readonly",
            ]

            creds = None
            if self._token_file and self._token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self._token_file), SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self._credentials_file), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                if self._token_file:
                    self._token_file.parent.mkdir(parents=True, exist_ok=True)
                    self._token_file.write_text(creds.to_json())

            self._service = build("calendar", "v3", credentials=creds)
            return self._service
        except Exception as e:
            logger.warning("Calendar service init failed, falling back to mock: %s", e)
            self._mock = True
            return None

    async def get_events(
        self,
        calendar_ids: list[str],
        target_date: Optional[date] = None,
        days_ahead: int = 1,
        timezone: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        """Fetch events for the given calendars over [target_date, target_date+days_ahead)."""
        if target_date is None:
            target_date = datetime.now(ZoneInfo(timezone)).date()

        if self._mock:
            return self._mock_events(target_date, timezone)

        tz = ZoneInfo(timezone)
        time_min = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=tz)
        time_max = time_min + timedelta(days=days_ahead)

        events = []
        service = self._get_service()
        if service is None:
            return self._mock_events(target_date, timezone)

        for cal_id in calendar_ids:
            try:
                result = service.events().list(
                    calendarId=cal_id,
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                ).execute()
                for item in result.get("items", []):
                    events.append(self._normalize_event(item, cal_id))
            except Exception as e:
                logger.error("Error fetching calendar %s: %s", cal_id, e)

        return sorted(events, key=lambda e: e["start"])

    def _normalize_event(self, raw: dict, calendar_id: str) -> dict[str, Any]:
        start = raw.get("start", {})
        end = raw.get("end", {})
        return {
            "id": raw.get("id", ""),
            "calendar_id": calendar_id,
            "summary": raw.get("summary", "No title"),
            "description": raw.get("description", ""),
            "location": raw.get("location", ""),
            "start": start.get("dateTime") or start.get("date", ""),
            "end": end.get("dateTime") or end.get("date", ""),
            "all_day": "date" in start and "dateTime" not in start,
            "attendees": [a.get("email", "") for a in raw.get("attendees", [])],
            "status": raw.get("status", "confirmed"),
        }

    def _mock_events(self, target_date: date, timezone: str) -> list[dict[str, Any]]:
        """Realistic mock events for demos and tests."""
        tz = ZoneInfo(timezone)
        base = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=tz)

        def t(hour: int, minute: int = 0) -> str:
            return (base + timedelta(hours=hour, minutes=minute)).isoformat()

        return [
            {
                "id": "mock_001",
                "calendar_id": "primary",
                "summary": "Henry — Soccer Practice",
                "description": "Outdoor field, Coach Davis",
                "location": "Riverside Park Field 3",
                "start": t(16, 0),
                "end": t(17, 30),
                "all_day": False,
                "attendees": [],
                "status": "confirmed",
            },
            {
                "id": "mock_002",
                "calendar_id": "primary",
                "summary": "Henry — Pediatrician Appointment",
                "description": "Annual checkup",
                "location": "Kids Health Clinic",
                "start": t(9, 0),
                "end": t(9, 45),
                "all_day": False,
                "attendees": [],
                "status": "tentative",
            },
            {
                "id": "mock_003",
                "calendar_id": "family",
                "summary": "Family Dinner",
                "description": "",
                "location": "Home",
                "start": t(18, 30),
                "end": t(19, 30),
                "all_day": False,
                "attendees": [],
                "status": "confirmed",
            },
        ]


def check_henry_constraints(event: dict[str, Any]) -> list[str]:
    """
    Check if an event for Henry violates scheduling constraints.

    Constraints:
    - no_early_morning_appts: No appointments before 9:00 AM
    - no_school_hours: No appointments 8:00 AM – 3:00 PM on weekdays
    """
    violations = []
    start_str = event.get("start", "")
    if not start_str or event.get("all_day"):
        return violations

    try:
        if "T" in start_str:
            start_dt = datetime.fromisoformat(start_str)
        else:
            return violations
    except ValueError:
        return violations

    hour = start_dt.hour
    weekday = start_dt.weekday()  # 0=Mon, 6=Sun

    if hour < 9:
        violations.append("⚠️ Early morning appointment (before 9 AM) — Henry may need to be up very early")

    if weekday < 5 and 8 <= hour < 15:  # weekday, school hours
        violations.append("⚠️ This falls during school hours (8 AM–3 PM on a weekday)")

    return violations
