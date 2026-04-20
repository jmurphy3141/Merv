"""Simple JSON-backed persistent memory store.

Stores:
- Family profile (members, preferences, constraints)
- Recent conversation summaries
- Rain-out history
- Note items for proactive surfacing

Phase 0 uses a flat JSON file. Phase 1 upgrades to vector + structured DB.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """Thread-safe(ish) JSON file memory for Phase 0."""

    DEFAULT_PROFILE: dict[str, Any] = {
        "family_members": ["Dad", "Mom", "Henry"],
        "notes": [],
        "rain_out_history": [],
        "conversation_summaries": [],
        "preferences": {
            "morning_brief_style": "concise",
            "timezone": "America/New_York",
        },
    }

    def __init__(self, memory_file: Optional[Path] = None):
        from merv.config.settings import get_settings
        settings = get_settings()
        self._file = memory_file or settings.memory_file
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                self._data = json.loads(self._file.read_text())
                logger.debug("Memory loaded from %s", self._file)
            except Exception as e:
                logger.warning("Failed to load memory file: %s — starting fresh", e)
                self._data = dict(self.DEFAULT_PROFILE)
        else:
            self._data = dict(self.DEFAULT_PROFILE)

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to save memory: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def add_note(self, note: str, category: str = "general") -> None:
        notes = self._data.setdefault("notes", [])
        notes.append({
            "text": note,
            "category": category,
            "created_at": datetime.utcnow().isoformat(),
        })
        self._save()

    def add_rain_out(self, event_name: str, date: str) -> None:
        history = self._data.setdefault("rain_out_history", [])
        history.append({"event": event_name, "date": date, "logged_at": datetime.utcnow().isoformat()})
        if len(history) > 50:
            history.pop(0)
        self._save()

    def add_conversation_summary(self, summary: str) -> None:
        summaries = self._data.setdefault("conversation_summaries", [])
        summaries.append({"summary": summary, "at": datetime.utcnow().isoformat()})
        if len(summaries) > 20:
            summaries.pop(0)
        self._save()

    def get_family_context_string(self) -> str:
        members = self._data.get("family_members", [])
        prefs = self._data.get("preferences", {})
        notes = self._data.get("notes", [])[-3:]  # last 3 notes
        lines = [
            f"Family members: {', '.join(members)}",
            f"Timezone: {prefs.get('timezone', 'America/New_York')}",
        ]
        if notes:
            lines.append("Recent notes: " + "; ".join(n["text"] for n in notes))
        return "\n".join(lines)
