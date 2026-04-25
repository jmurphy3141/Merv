"""Coding skill — wraps the ReAct coding agent as a Merv skill."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic

from merv.agents.coding_agent import run_coding_task
from merv.config.settings import get_settings
from merv.skills.base import BaseSkill

if TYPE_CHECKING:
    from merv.core.state import MervState

logger = logging.getLogger(__name__)


class CodingSkill(BaseSkill):
    """Writes, runs, and tests Python code using a LangGraph ReAct agent loop."""

    name = "coding"
    description = "Writes and tests Python code iteratively using tool-calling."

    def __init__(self, work_dir: str | None = None):
        super().__init__()
        self._work_dir = work_dir

    async def run(self, state: "MervState") -> str:
        settings = get_settings()

        if not settings.anthropic_api_key:
            return (
                "I need an Anthropic API key to run the coding agent. "
                "Please set ANTHROPIC_API_KEY in your .env file."
            )

        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
        )

        task = state.current_input
        self.logger.info("Running coding agent for task: %.80s...", task)

        return await run_coding_task(
            task=task,
            llm=llm,
            work_dir=self._work_dir,
        )
