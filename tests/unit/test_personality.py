"""Unit tests for Merv's personality/system prompt."""

import pytest
from merv.core.personality import get_system_prompt, MERV_SYSTEM_PROMPT


class TestPersonality:
    def test_system_prompt_is_non_empty(self):
        assert len(MERV_SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_family_priority(self):
        assert "Family" in MERV_SYSTEM_PROMPT or "family" in MERV_SYSTEM_PROMPT

    def test_system_prompt_mentions_henry(self):
        assert "Henry" in MERV_SYSTEM_PROMPT

    def test_get_system_prompt_no_extra(self):
        prompt = get_system_prompt()
        assert prompt == MERV_SYSTEM_PROMPT

    def test_get_system_prompt_with_extra_context(self):
        prompt = get_system_prompt(extra_context="Today is Monday. Rain risk: high.")
        assert "Today is Monday" in prompt
        assert MERV_SYSTEM_PROMPT in prompt

    def test_prompt_mentions_priority_order(self):
        assert "1." in MERV_SYSTEM_PROMPT or "first" in MERV_SYSTEM_PROMPT.lower()
