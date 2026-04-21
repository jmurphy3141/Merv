"""Unit tests for the intent classification router."""

import pytest
from merv.core.intent_router import fast_classify, classify_intent
from merv.core.state import IntentType


class TestFastClassify:
    """Tests for keyword-based fast classification."""

    @pytest.mark.parametrize("text,expected_intent", [
        ("Good morning! What's on today?", IntentType.MORNING_ROUTINE),
        ("morning brief please", IntentType.MORNING_ROUTINE),
        ("What's up today?", IntentType.MORNING_ROUTINE),
        ("What's on my calendar?", IntentType.FAMILY_ASSISTANT),
        ("Henry has soccer practice today", IntentType.FAMILY_ASSISTANT),
        ("Will it rain tomorrow for the baseball game?", IntentType.FAMILY_ASSISTANT),
        ("Check my emails", IntentType.FAMILY_ASSISTANT),
        ("What's in my inbox?", IntentType.FAMILY_ASSISTANT),
        ("Schedule a doctor appointment for Henry", IntentType.FAMILY_ASSISTANT),
        ("Any tasks due today?", IntentType.PROJECT_MANAGER),
        ("Project deadline reminder", IntentType.PROJECT_MANAGER),
        ("Fix the bug in the auth module", IntentType.ENGINEERING),
        ("Create a pull request for the feature", IntentType.ENGINEERING),
        ("Hello there!", IntentType.CONVERSATION),
    ])
    def test_classify_known_intents(self, text, expected_intent):
        intent, confidence = fast_classify(text)
        assert intent == expected_intent

    def test_high_confidence_on_keyword_match(self):
        _, confidence = fast_classify("morning brief")
        assert confidence >= 0.8

    def test_lower_confidence_on_no_match(self):
        _, confidence = fast_classify("Hello, how are you?")
        assert confidence < 0.8

    def test_morning_takes_priority_over_family(self):
        """'morning' keyword should route to morning_routine, not family."""
        intent, _ = fast_classify("Good morning, what's on the family calendar?")
        assert intent == IntentType.MORNING_ROUTINE

    def test_case_insensitive(self):
        intent, _ = fast_classify("HENRY HAS SOCCER PRACTICE")
        assert intent == IntentType.FAMILY_ASSISTANT


class TestClassifyIntent:
    """Tests for full classify_intent (no LLM — fast path only)."""

    @pytest.mark.asyncio
    async def test_classify_without_llm(self):
        intent, confidence = await classify_intent("morning brief", llm=None)
        assert intent == IntentType.MORNING_ROUTINE
        assert confidence >= 0.8

    @pytest.mark.asyncio
    async def test_classify_conversation_without_llm(self):
        intent, confidence = await classify_intent("Tell me a joke", llm=None)
        assert intent == IntentType.CONVERSATION
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_all_intents_reachable(self):
        """Smoke test: every IntentType except UNKNOWN is reachable from text."""
        test_pairs = [
            ("morning brief", IntentType.MORNING_ROUTINE),
            ("calendar today", IntentType.FAMILY_ASSISTANT),
            ("project task deadline", IntentType.PROJECT_MANAGER),
            ("fix bug in code", IntentType.ENGINEERING),
            ("hello", IntentType.CONVERSATION),
        ]
        for text, expected in test_pairs:
            intent, _ = await classify_intent(text, llm=None)
            assert intent == expected, f"'{text}' → {intent}, expected {expected}"
