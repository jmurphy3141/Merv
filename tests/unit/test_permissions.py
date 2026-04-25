"""Unit tests for the coding agent permission system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merv.agents.permissions import (
    PERMISSION_INFO,
    Permission,
    PermissionStore,
    is_reset,
    parse_deny,
    parse_grant,
)

ALL_PERMS = list(Permission)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "coding_permissions.json"


@pytest.fixture
def store(store_path: Path) -> PermissionStore:
    return PermissionStore(store_path)


# ── PermissionStore basics ────────────────────────────────────────────────────

class TestPermissionStoreGrant:
    def test_new_store_has_nothing_granted(self, store):
        for perm in Permission:
            assert not store.is_granted(perm)

    def test_grant_marks_as_granted(self, store):
        store.grant(Permission.WRITE_FILES)
        assert store.is_granted(Permission.WRITE_FILES)

    def test_grant_saves_to_disk(self, store, store_path):
        store.grant(Permission.EXECUTE_CODE)
        raw = json.loads(store_path.read_text())
        assert raw["permissions"]["execute_code"]["status"] == "granted"

    def test_grant_records_timestamp(self, store, store_path):
        store.grant(Permission.RUN_TESTS)
        raw = json.loads(store_path.read_text())
        assert "granted_at" in raw["permissions"]["run_tests"]

    def test_grant_records_label_and_reason(self, store, store_path):
        store.grant(Permission.READ_FILES)
        raw = json.loads(store_path.read_text())
        entry = raw["permissions"]["read_files"]
        assert entry["label"] == PERMISSION_INFO[Permission.READ_FILES]["label"]
        assert entry["reason"] == PERMISSION_INFO[Permission.READ_FILES]["reason"]

    def test_deny_marks_as_not_granted(self, store):
        store.deny(Permission.EXECUTE_CODE)
        assert not store.is_granted(Permission.EXECUTE_CODE)

    def test_deny_saves_to_disk(self, store, store_path):
        store.deny(Permission.WRITE_FILES)
        raw = json.loads(store_path.read_text())
        assert raw["permissions"]["write_files"]["status"] == "denied"

    def test_reset_clears_all(self, store):
        store.grant(Permission.WRITE_FILES)
        store.grant(Permission.EXECUTE_CODE)
        store.reset()
        assert store.missing_permissions() == ALL_PERMS

    def test_reset_saves_empty_to_disk(self, store, store_path):
        store.grant(Permission.WRITE_FILES)
        store.reset()
        raw = json.loads(store_path.read_text())
        assert raw["permissions"] == {}


class TestMissingPermissions:
    def test_all_missing_initially(self, store):
        assert set(store.missing_permissions()) == set(ALL_PERMS)

    def test_granted_not_in_missing(self, store):
        store.grant(Permission.WRITE_FILES)
        missing = store.missing_permissions()
        assert Permission.WRITE_FILES not in missing
        assert len(missing) == len(ALL_PERMS) - 1

    def test_denied_still_in_missing(self, store):
        store.deny(Permission.EXECUTE_CODE)
        assert Permission.EXECUTE_CODE in store.missing_permissions()

    def test_all_granted_means_no_missing(self, store):
        for perm in Permission:
            store.grant(perm)
        assert store.missing_permissions() == []

    def test_denied_permissions_list(self, store):
        store.deny(Permission.RUN_TESTS)
        assert Permission.RUN_TESTS in store.denied_permissions()
        assert Permission.WRITE_FILES not in store.denied_permissions()


class TestPendingTask:
    def test_no_pending_task_initially(self, store):
        assert store.get_pending_task() is None

    def test_set_and_get_pending_task(self, store):
        store.set_pending_task("write a fibonacci function")
        assert store.get_pending_task() == "write a fibonacci function"

    def test_pending_task_persisted_to_disk(self, store, store_path):
        store.set_pending_task("sort a list")
        raw = json.loads(store_path.read_text())
        assert raw["pending_task"] == "sort a list"

    def test_clear_pending_task(self, store):
        store.set_pending_task("do something")
        store.clear_pending_task()
        assert store.get_pending_task() is None

    def test_reset_clears_pending_task(self, store):
        store.set_pending_task("some task")
        store.reset()
        assert store.get_pending_task() is None


class TestPersistence:
    def test_loads_existing_data_on_init(self, store_path):
        # Write grants manually
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps({
            "permissions": {
                "write_files": {"status": "granted", "granted_at": "2026-01-01"},
            },
            "pending_task": "existing task",
        }))

        loaded = PermissionStore(store_path)
        assert loaded.is_granted(Permission.WRITE_FILES)
        assert not loaded.is_granted(Permission.EXECUTE_CODE)
        assert loaded.get_pending_task() == "existing task"

    def test_corrupt_file_starts_fresh(self, store_path):
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("{{broken json")
        store = PermissionStore(store_path)
        assert store.missing_permissions() == ALL_PERMS

    def test_missing_file_starts_fresh(self, tmp_path):
        store = PermissionStore(tmp_path / "nonexistent.json")
        assert store.missing_permissions() == ALL_PERMS


# ── Formatting ────────────────────────────────────────────────────────────────

class TestFormatRequest:
    def test_lists_all_missing_permissions(self, store):
        text = store.format_request(ALL_PERMS)
        for perm in Permission:
            assert PERMISSION_INFO[perm]["label"] in text
            assert PERMISSION_INFO[perm]["reason"] in text

    def test_includes_allow_instruction(self, store):
        text = store.format_request(ALL_PERMS)
        assert "allow" in text.lower()

    def test_includes_scope_for_each(self, store):
        text = store.format_request(ALL_PERMS)
        for perm in Permission:
            assert PERMISSION_INFO[perm]["scope"] in text

    def test_single_missing_permission(self, store):
        store.grant(Permission.READ_FILES)
        store.grant(Permission.EXECUTE_CODE)
        store.grant(Permission.RUN_TESTS)
        missing = store.missing_permissions()
        text = store.format_request(missing)
        assert PERMISSION_INFO[Permission.WRITE_FILES]["label"] in text
        assert PERMISSION_INFO[Permission.READ_FILES]["label"] not in text

    def test_grant_confirmation_lists_names(self, store):
        text = store.format_grant_confirmation([Permission.WRITE_FILES, Permission.EXECUTE_CODE])
        assert "Write files" in text
        assert "Execute Python scripts" in text

    def test_deny_confirmation_includes_reset_hint(self, store):
        text = store.format_deny_confirmation([Permission.RUN_TESTS])
        assert "reset" in text.lower()


# ── Parse helpers ─────────────────────────────────────────────────────────────

class TestParseGrant:
    def test_allow_returns_all_missing(self):
        missing = list(Permission)
        result = parse_grant("allow", missing)
        assert set(result) == set(missing)

    def test_yes_returns_all_missing(self):
        missing = [Permission.WRITE_FILES]
        result = parse_grant("yes", missing)
        assert result == [Permission.WRITE_FILES]

    def test_grant_specific_permission(self):
        missing = list(Permission)
        result = parse_grant("allow write_files execute_code", missing)
        assert Permission.WRITE_FILES in result
        assert Permission.EXECUTE_CODE in result
        assert Permission.RUN_TESTS not in result

    def test_not_a_grant_message_returns_none(self):
        assert parse_grant("write me a function", list(Permission)) is None
        assert parse_grant("what is the weather?", list(Permission)) is None

    def test_case_insensitive(self):
        result = parse_grant("ALLOW", list(Permission))
        assert result is not None

    def test_approve_keyword(self):
        result = parse_grant("I approve", list(Permission))
        assert result is not None


class TestParseDeny:
    def test_deny_returns_all_missing(self):
        missing = list(Permission)
        result = parse_deny("deny", missing)
        assert set(result) == set(missing)

    def test_no_returns_all_missing(self):
        missing = [Permission.EXECUTE_CODE]
        result = parse_deny("no", missing)
        assert result == [Permission.EXECUTE_CODE]

    def test_deny_specific_permission(self):
        missing = list(Permission)
        result = parse_deny("deny execute_code", missing)
        assert Permission.EXECUTE_CODE in result
        assert Permission.WRITE_FILES not in result

    def test_not_a_deny_message_returns_none(self):
        assert parse_deny("write me a function", list(Permission)) is None
        assert parse_deny("allow it", list(Permission)) is None


class TestIsReset:
    def test_reset_permissions(self):
        assert is_reset("reset coding permissions")
        assert is_reset("reset permissions")

    def test_not_reset(self):
        assert not is_reset("write a function")
        assert not is_reset("allow")
