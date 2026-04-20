"""Base class for all Merv skills."""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merv.core.state import MervState


class BaseSkill(abc.ABC):
    """All skills inherit from this. Skills are synchronous-safe but expose async run()."""

    name: str = "base"
    description: str = ""

    def __init__(self):
        self.logger = logging.getLogger(f"merv.skills.{self.name}")

    @abc.abstractmethod
    async def run(self, state: "MervState") -> str:
        """Execute the skill and return a formatted response string."""
        ...
