"""Permission system for the coding agent.

Before the agent can write files or execute code it must have explicit user
approval. Decisions are stored in a JSON file so the user is only asked once.

Flow:
  1. User asks for a coding task.
  2. CodingSkill checks for missing permissions and returns a formatted request,
     storing the original task as a pending task.
  3. User replies "allow" (or "deny").
  4. CodingSkill grants/denies, then runs the stored pending task automatically.

Users can say "reset coding permissions" at any time to start over.
"""

from __future__ import annotations

import enum
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path.home() / ".config" / "merv" / "data" / "coding_permissions.json"


class Permission(str, enum.Enum):
    WRITE_FILES = "write_files"
    READ_FILES = "read_files"
    EXECUTE_CODE = "execute_code"
    RUN_TESTS = "run_tests"


PERMISSION_INFO: dict[Permission, dict[str, str]] = {
    Permission.WRITE_FILES: {
        "label": "Write files",
        "reason": "The agent saves generated code to disk so it can be executed and tested.",
        "scope": "Confined to the coding workspace (/tmp/merv_coding/ by default).",
    },
    Permission.READ_FILES: {
        "label": "Read files",
        "reason": "The agent reads its own files to review and fix errors.",
        "scope": "Confined to the coding workspace — cannot read outside that directory.",
    },
    Permission.EXECUTE_CODE: {
        "label": "Execute Python scripts",
        "reason": "The agent runs generated scripts to verify they produce the expected output.",
        "scope": "30-second timeout, no shell=True, no network access from the agent itself.",
    },
    Permission.RUN_TESTS: {
        "label": "Run pytest",
        "reason": "The agent runs your test suite to verify correctness and iterate on failures.",
        "scope": "30-second timeout, confined to the coding workspace.",
    },
}

# ── Grant / deny parsing ───────────────────────────────────────────────────────

_ALLOW_RE = re.compile(r"\b(allow|yes|grant|approve|ok|sure|go ahead|proceed|yep|yup)\b", re.I)
_DENY_RE = re.compile(r"\b(deny|no|refuse|reject|block|cancel|nope|nah)\b", re.I)
_RESET_RE = re.compile(r"\breset\s+(coding\s+)?permissions?\b", re.I)

_PERM_BY_NAME: dict[str, Permission] = {p.value: p for p in Permission}


def parse_grant(text: str, missing: list[Permission]) -> list[Permission] | None:
    """Return permissions to grant from user text, or None if not a grant message.

    If specific permission names are mentioned they are returned; otherwise all
    missing permissions are returned (the user approved everything).
    """
    if not _ALLOW_RE.search(text):
        return None
    named = [p for name, p in _PERM_BY_NAME.items() if name in text.lower()]
    return named if named else list(missing)


def parse_deny(text: str, missing: list[Permission]) -> list[Permission] | None:
    """Return permissions to deny from user text, or None if not a deny message."""
    if not _DENY_RE.search(text):
        return None
    named = [p for name, p in _PERM_BY_NAME.items() if name in text.lower()]
    return named if named else list(missing)


def is_reset(text: str) -> bool:
    return bool(_RESET_RE.search(text))


# ── PermissionStore ────────────────────────────────────────────────────────────

class PermissionStore:
    """JSON-backed store for coding agent permission decisions.

    Schema:
        {
            "permissions": {
                "write_files": {"status": "granted", "granted_at": "...", ...},
                ...
            },
            "pending_task": "write a fibonacci function" | null
        }
    """

    def __init__(self, store_path: Optional[Path] = None):
        self._path = store_path or _DEFAULT_STORE_PATH
        self._data: dict = {"permissions": {}, "pending_task": None}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw if isinstance(raw, dict) else self._data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load permission store (%s) — starting fresh.", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not save permission store: %s", exc)

    # ── Permission operations ──────────────────────────────────────────────────

    def is_granted(self, perm: Permission) -> bool:
        return self._data["permissions"].get(perm.value, {}).get("status") == "granted"

    def grant(self, perm: Permission) -> None:
        self._data["permissions"][perm.value] = {
            "status": "granted",
            "label": PERMISSION_INFO[perm]["label"],
            "reason": PERMISSION_INFO[perm]["reason"],
            "granted_at": datetime.utcnow().isoformat(),
        }
        self._save()

    def deny(self, perm: Permission) -> None:
        self._data["permissions"][perm.value] = {
            "status": "denied",
            "label": PERMISSION_INFO[perm]["label"],
            "reason": PERMISSION_INFO[perm]["reason"],
            "denied_at": datetime.utcnow().isoformat(),
        }
        self._save()

    def reset(self) -> None:
        self._data = {"permissions": {}, "pending_task": None}
        self._save()

    def missing_permissions(self) -> list[Permission]:
        """Return all permissions that have not been explicitly granted."""
        return [p for p in Permission if not self.is_granted(p)]

    def denied_permissions(self) -> list[Permission]:
        return [
            p for p in Permission
            if self._data["permissions"].get(p.value, {}).get("status") == "denied"
        ]

    # ── Pending task ───────────────────────────────────────────────────────────

    def set_pending_task(self, task: str) -> None:
        self._data["pending_task"] = task
        self._save()

    def get_pending_task(self) -> str | None:
        return self._data.get("pending_task")

    def clear_pending_task(self) -> None:
        self._data["pending_task"] = None
        self._save()

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_request(self, missing: list[Permission]) -> str:
        lines = [
            "Before I can run this coding task, I need your approval for the following:\n"
        ]
        for i, perm in enumerate(missing, 1):
            info = PERMISSION_INFO[perm]
            lines.append(f"{i}. **{info['label']}** (`{perm.value}`)")
            lines.append(f"   *Why:* {info['reason']}")
            lines.append(f"   *Scope:* {info['scope']}")
            lines.append("")

        lines += [
            "Reply **'allow'** to grant all, or name specific ones:",
            "  • `allow write_files execute_code`",
            "  • `deny run_tests` (to block a specific one)",
            "",
            "Say **'reset coding permissions'** at any time to start over.",
            "Your choices are saved — you won't be asked again.",
        ]
        return "\n".join(lines)

    def format_grant_confirmation(self, granted: list[Permission]) -> str:
        names = ", ".join(PERMISSION_INFO[p]["label"] for p in granted)
        return f"Permissions granted: {names}."

    def format_deny_confirmation(self, denied: list[Permission]) -> str:
        names = ", ".join(PERMISSION_INFO[p]["label"] for p in denied)
        return (
            f"Permissions denied: {names}. "
            "Say 'reset coding permissions' to change this later."
        )
