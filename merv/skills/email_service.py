"""Gmail integration — fetch and triage emails for the morning brief.

Uses Gmail API with readonly scope. Falls back to realistic mock data when
credentials are unavailable (dev/test mode).
"""

from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Priority keywords that bump an email to the top of the brief
_URGENT_PATTERNS = re.compile(
    r"\b(urgent|asap|action required|time.sensitive|canceled|cancelled|emergency|important)\b",
    re.I,
)
_FAMILY_PATTERNS = re.compile(
    r"\b(henry|soccer|baseball|lacrosse|school|pta|coach|pediatrician|appointment|doctor)\b",
    re.I,
)


class GmailService:
    """
    Fetch recent emails and perform priority triage for the morning brief.
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
        if self._service:
            return self._service
        if self._mock:
            return None

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
            creds = None
            if self._token_file and self._token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self._token_file), SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    logger.warning("Gmail credentials invalid and no refresh token available")
                    self._mock = True
                    return None

            self._service = build("gmail", "v1", credentials=creds)
            return self._service
        except Exception as e:
            logger.warning("Gmail service init failed, falling back to mock: %s", e)
            self._mock = True
            return None

    async def get_recent_emails(
        self,
        max_results: int = 10,
        hours_back: int = 16,
        label_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch emails received in the last `hours_back` hours.
        Returns a list of normalized email dicts, sorted by priority.
        """
        if self._mock:
            return self._mock_emails()

        service = self._get_service()
        if service is None:
            return self._mock_emails()

        try:
            after_ts = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())
            query = f"after:{after_ts}"
            if label_ids:
                query += " " + " ".join(f"label:{l}" for l in label_ids)

            result = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=max_results,
            ).execute()

            messages = []
            for msg_ref in result.get("messages", []):
                try:
                    msg = service.users().messages().get(
                        userId="me",
                        id=msg_ref["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date"],
                    ).execute()
                    messages.append(self._normalize_email(msg))
                except Exception as e:
                    logger.warning("Error fetching email %s: %s", msg_ref["id"], e)

            return self._triage(messages)
        except Exception as e:
            logger.error("Error fetching emails: %s", e)
            return self._mock_emails()

    def _normalize_email(self, raw: dict) -> dict[str, Any]:
        headers = {h["name"]: h["value"] for h in raw.get("payload", {}).get("headers", [])}
        snippet = raw.get("snippet", "")
        subject = headers.get("Subject", "(no subject)")
        sender = headers.get("From", "Unknown")
        return {
            "id": raw.get("id", ""),
            "subject": subject,
            "sender": sender,
            "snippet": snippet,
            "date": headers.get("Date", ""),
            "labels": raw.get("labelIds", []),
            "urgent": bool(_URGENT_PATTERNS.search(subject) or _URGENT_PATTERNS.search(snippet)),
            "family_related": bool(_FAMILY_PATTERNS.search(subject) or _FAMILY_PATTERNS.search(snippet)),
        }

    def _triage(self, emails: list[dict]) -> list[dict]:
        """Sort: urgent first, then family-related, then the rest."""
        urgent = [e for e in emails if e["urgent"]]
        family = [e for e in emails if e["family_related"] and not e["urgent"]]
        rest = [e for e in emails if not e["urgent"] and not e["family_related"]]
        return urgent + family + rest

    def _mock_emails(self) -> list[dict[str, Any]]:
        """Realistic mock emails for demos and tests."""
        return [
            {
                "id": "mock_email_001",
                "subject": "⚠️ Soccer Practice CANCELED — Rain tomorrow",
                "sender": "Coach Davis <coach.davis@soccer.org>",
                "snippet": "Due to weather forecast, practice at Riverside Park is canceled for tomorrow. Please check for a makeup date.",
                "date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
                "labels": ["INBOX"],
                "urgent": True,
                "family_related": True,
            },
            {
                "id": "mock_email_002",
                "subject": "Henry's appointment reminder — Dr. Kim",
                "sender": "Kids Health Clinic <noreply@kidshealthclinic.com>",
                "snippet": "This is a reminder for Henry's annual checkup tomorrow at 9:00 AM.",
                "date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
                "labels": ["INBOX"],
                "urgent": False,
                "family_related": True,
            },
            {
                "id": "mock_email_003",
                "subject": "AWS invoice — April 2026",
                "sender": "billing@aws.amazon.com",
                "snippet": "Your April 2026 invoice is available. Amount due: $142.50.",
                "date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"),
                "labels": ["INBOX"],
                "urgent": False,
                "family_related": False,
            },
        ]
