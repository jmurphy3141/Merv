"""Intent classification — routes user input to the right skill or sub-agent.

Uses a lightweight keyword/pattern pass first (fast, no LLM call needed for
obvious requests), then falls back to LLM classification for ambiguous input.
"""

from __future__ import annotations

import re
from typing import Optional

from merv.core.state import IntentType

# ── Fast keyword routing (no LLM) ─────────────────────────────────────────────
_PATTERNS: list[tuple[re.Pattern, IntentType]] = [
    # Morning routine triggers
    (re.compile(r"\b(morning|good morning|daily brief|what'?s? (up|today|on today))\b", re.I), IntentType.MORNING_ROUTINE),
    # Family / calendar
    (re.compile(r"\b(calendar|schedule|appointment|event|henry|soccer|baseball|lacrosse|rain|weather|family|mom|dad)\b", re.I), IntentType.FAMILY_ASSISTANT),
    # Email
    (re.compile(r"\b(email|emails|inbox|gmail|mail|message)\b", re.I), IntentType.FAMILY_ASSISTANT),
    # Project management
    (re.compile(r"\b(project|tasks?|todo|deadline|milestone|standup|sprint|backlog)\b", re.I), IntentType.PROJECT_MANAGER),
    # Engineering / coding
    (re.compile(
        r"\b(code|coding|bugs?|refactor|deploy|tests?|testing|build|pr|pull request|commit|debug"
        r"|script|function|implement|algorithm|python|pytest|unittest|class|module)\b",
        re.I,
    ), IntentType.ENGINEERING),
]


def fast_classify(text: str) -> tuple[IntentType, float]:
    """
    Pattern-based classification — runs in microseconds.
    Returns (intent, confidence). Confidence is 0.8 for keyword hits, lower otherwise.
    """
    for pattern, intent in _PATTERNS:
        if pattern.search(text):
            return intent, 0.8
    return IntentType.CONVERSATION, 0.5


async def classify_intent(text: str, llm=None) -> tuple[IntentType, float]:
    """
    Full classification: fast path first, LLM fallback for ambiguous input.
    Pass llm=None to skip the LLM fallback (useful in tests).
    """
    intent, confidence = fast_classify(text)

    # High-confidence fast hit — skip LLM
    if confidence >= 0.8:
        return intent, confidence

    # LLM fallback for low-confidence cases
    if llm is None:
        return intent, confidence

    prompt = f"""\
Classify this user message into exactly one category:
- morning_routine: morning brief, daily summary, what's today
- family_assistant: calendar, schedule, appointments, family logistics, weather impacts on activities
- project_manager: project tasks, deadlines, milestones
- engineering: code, bugs, deployments, pull requests
- conversation: general chat, questions, everything else

Message: "{text}"

Reply with ONLY the category name, nothing else."""

    try:
        response = await llm.ainvoke(prompt)
        raw = response.content.strip().lower()
        intent_map = {i.value: i for i in IntentType}
        if raw in intent_map:
            return intent_map[raw], 0.9
    except Exception:
        pass  # fall through to keyword result

    return intent, confidence
