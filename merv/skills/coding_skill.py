"""Coding skill — wraps the ReAct coding agent as a Merv skill.

Permission flow:
  1. First coding request → PermissionStore has nothing granted → returns a
     formatted permission request explaining each permission and why it's needed,
     and stores the original task as a pending task.
  2. User replies "allow" (or similar) → permissions are granted, the stored
     pending task is run automatically, result returned.
  3. Subsequent coding requests → all permissions already granted → agent runs
     immediately with no prompts.

Users can say "reset coding permissions" to revoke all grants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic

from merv.agents.coding_agent import run_coding_task
from merv.agents.permissions import (
    Permission,
    PermissionStore,
    is_reset,
    parse_deny,
    parse_grant,
)
from merv.config.settings import get_settings
from merv.skills.base import BaseSkill

if TYPE_CHECKING:
    from merv.core.state import MervState

logger = logging.getLogger(__name__)


class CodingSkill(BaseSkill):
    """Writes and tests Python code iteratively using a LangGraph ReAct agent loop."""

    name = "coding"
    description = "Writes and tests Python code iteratively using tool-calling."

    def __init__(self, work_dir: str | None = None, _store_path=None):
        super().__init__()
        self._work_dir = work_dir
        self._store_path = _store_path  # injectable for tests

    def _build_perm_store(self, settings) -> PermissionStore:
        path = self._store_path or (settings.data_dir / "coding_permissions.json")
        return PermissionStore(path)

    def _build_llm(self, settings):
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
        )

    async def run(self, state: "MervState") -> str:
        settings = get_settings()

        if not settings.anthropic_api_key:
            return (
                "I need an Anthropic API key to run the coding agent. "
                "Please set ANTHROPIC_API_KEY in your .env file."
            )

        perm_store = self._build_perm_store(settings)
        user_input = state.current_input.strip()

        # ── Reset request ──────────────────────────────────────────────────────
        if is_reset(user_input):
            perm_store.reset()
            return (
                "Coding permissions have been reset. "
                "I'll ask for your approval again on the next coding task."
            )

        missing = perm_store.missing_permissions()

        # ── Grant response ─────────────────────────────────────────────────────
        to_grant = parse_grant(user_input, missing)
        if to_grant is not None and missing:
            for perm in to_grant:
                perm_store.grant(perm)
            self.logger.info("Granted permissions: %s", [p.value for p in to_grant])

            still_missing = perm_store.missing_permissions()
            if still_missing:
                return perm_store.format_request(still_missing)

            confirmation = perm_store.format_grant_confirmation(to_grant)

            # Run the task the user originally asked for, if stored
            pending = perm_store.get_pending_task()
            perm_store.clear_pending_task()
            if pending:
                self.logger.info("Running pending task after grant: %.80s...", pending)
                llm = self._build_llm(settings)
                result = await run_coding_task(
                    task=pending,
                    llm=llm,
                    work_dir=self._work_dir,
                    permission_store=perm_store,
                )
                return f"{confirmation}\n\nNow running your task...\n\n{result}"

            return f"{confirmation} What would you like me to code?"

        # ── Deny response ──────────────────────────────────────────────────────
        to_deny = parse_deny(user_input, missing)
        if to_deny is not None and missing:
            for perm in to_deny:
                perm_store.deny(perm)
            perm_store.clear_pending_task()
            self.logger.info("Denied permissions: %s", [p.value for p in to_deny])
            return perm_store.format_deny_confirmation(to_deny)

        # ── Permission request (first time or after denial) ────────────────────
        if missing:
            perm_store.set_pending_task(user_input)
            return perm_store.format_request(missing)

        # ── All granted — run the agent ────────────────────────────────────────
        self.logger.info("Running coding agent for task: %.80s...", user_input)
        llm = self._build_llm(settings)
        return await run_coding_task(
            task=user_input,
            llm=llm,
            work_dir=self._work_dir,
            permission_store=perm_store,
        )
