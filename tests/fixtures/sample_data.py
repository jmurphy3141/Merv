"""Shared test data fixtures used across unit and integration tests."""

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")
TODAY = date(2026, 4, 20)  # Monday


def make_event(
    summary: str,
    hour: int = 10,
    minute: int = 0,
    duration_hours: float = 1.0,
    status: str = "confirmed",
    target_date: date = TODAY,
    all_day: bool = False,
) -> dict:
    base = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=TZ)
    start_dt = base + timedelta(hours=hour, minutes=minute)
    end_dt = start_dt + timedelta(hours=duration_hours)
    if all_day:
        return {
            "id": f"test_{summary[:10]}",
            "calendar_id": "primary",
            "summary": summary,
            "description": "",
            "location": "",
            "start": target_date.isoformat(),
            "end": (target_date + timedelta(days=1)).isoformat(),
            "all_day": True,
            "attendees": [],
            "status": status,
        }
    return {
        "id": f"test_{summary[:10]}",
        "calendar_id": "primary",
        "summary": summary,
        "description": "",
        "location": "",
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "all_day": False,
        "attendees": [],
        "status": status,
    }


SAMPLE_EVENTS_RAINY_DAY = [
    make_event("Henry — Soccer Practice", hour=16, target_date=TODAY),
    make_event("Henry — Pediatrician Appointment", hour=9, status="tentative", target_date=TODAY),
    make_event("Family Dinner", hour=18, target_date=TODAY),
]

SAMPLE_EVENTS_CLEAR_DAY = [
    make_event("Henry — Soccer Practice", hour=16, target_date=TODAY),
    make_event("Family Dinner", hour=18, target_date=TODAY),
]

SAMPLE_EVENTS_SCHOOL_HOUR_CONFLICT = [
    make_event("Henry — Doctor Appointment", hour=10, target_date=TODAY),  # school hours violation
]

SAMPLE_EMAILS_WITH_CANCELLATION = [
    {
        "id": "e001",
        "subject": "Soccer Practice CANCELED — Rain",
        "sender": "Coach Davis <coach@soccer.org>",
        "snippet": "Practice canceled due to rain forecast.",
        "date": "Mon, 20 Apr 2026 08:00:00 +0000",
        "labels": ["INBOX"],
        "urgent": True,
        "family_related": True,
    },
    {
        "id": "e002",
        "subject": "Henry appointment reminder",
        "sender": "clinic@health.com",
        "snippet": "Reminder: annual checkup tomorrow at 9 AM.",
        "date": "Mon, 20 Apr 2026 07:30:00 +0000",
        "labels": ["INBOX"],
        "urgent": False,
        "family_related": True,
    },
]

RAINY_WEATHER = {
    "date": TODAY.isoformat(),
    "temp_max": 54.0,
    "temp_min": 46.0,
    "weather_code": 61,
    "precipitation_mm": 18.0,
    "rain_risk": True,
    "description": "Light rain expected",
}

CLEAR_WEATHER = {
    "date": TODAY.isoformat(),
    "temp_max": 72.0,
    "temp_min": 58.0,
    "weather_code": 1,
    "precipitation_mm": 0.0,
    "rain_risk": False,
    "description": "Mainly clear",
}
