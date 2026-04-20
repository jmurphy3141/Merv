"""Unit tests for the Gmail email service and triage logic."""

import pytest

from merv.skills.email_service import GmailService


class TestGmailServiceMock:
    def setup_method(self):
        self.service = GmailService(mock=True)

    @pytest.mark.asyncio
    async def test_returns_list_of_emails(self):
        emails = await self.service.get_recent_emails()
        assert isinstance(emails, list)
        assert len(emails) > 0

    @pytest.mark.asyncio
    async def test_email_has_required_fields(self):
        emails = await self.service.get_recent_emails()
        required = {"id", "subject", "sender", "snippet", "urgent", "family_related"}
        for email in emails:
            assert required.issubset(email.keys())

    @pytest.mark.asyncio
    async def test_urgent_emails_come_first(self):
        emails = await self.service.get_recent_emails()
        urgent_indices = [i for i, e in enumerate(emails) if e["urgent"]]
        non_urgent_indices = [i for i, e in enumerate(emails) if not e["urgent"]]
        if urgent_indices and non_urgent_indices:
            assert max(urgent_indices) < min(non_urgent_indices)

    @pytest.mark.asyncio
    async def test_cancellation_email_is_urgent(self):
        emails = await self.service.get_recent_emails()
        cancellation = next(
            (e for e in emails if "canceled" in e["subject"].lower() or "cancelled" in e["subject"].lower()),
            None,
        )
        if cancellation:
            assert cancellation["urgent"] is True

    @pytest.mark.asyncio
    async def test_family_emails_come_before_other(self):
        emails = await self.service.get_recent_emails()
        family_indices = [i for i, e in enumerate(emails) if e["family_related"] and not e["urgent"]]
        other_indices = [i for i, e in enumerate(emails) if not e["family_related"] and not e["urgent"]]
        if family_indices and other_indices:
            assert max(family_indices) < min(other_indices)

    def test_triage_ordering(self):
        """Test the _triage method directly."""
        test_emails = [
            {"subject": "Regular", "urgent": False, "family_related": False},
            {"subject": "Family", "urgent": False, "family_related": True},
            {"subject": "URGENT", "urgent": True, "family_related": False},
        ]
        result = self.service._triage(test_emails)
        assert result[0]["subject"] == "URGENT"
        assert result[1]["subject"] == "Family"
        assert result[2]["subject"] == "Regular"
