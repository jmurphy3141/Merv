"""Unit tests for the MemoryStore."""

import pytest
import json
import tempfile
from pathlib import Path

from merv.memory.store import MemoryStore


@pytest.fixture
def temp_memory_store(tmp_path):
    memory_file = tmp_path / "test_memory.json"
    return MemoryStore(memory_file=memory_file)


class TestMemoryStore:
    def test_default_data_initialized(self, temp_memory_store):
        store = temp_memory_store
        members = store.get("family_members")
        assert isinstance(members, list)
        assert len(members) > 0

    def test_set_and_get_value(self, temp_memory_store):
        store = temp_memory_store
        store.set("test_key", "test_value")
        assert store.get("test_key") == "test_value"

    def test_get_missing_key_returns_default(self, temp_memory_store):
        store = temp_memory_store
        assert store.get("nonexistent", default="fallback") == "fallback"

    def test_add_note(self, temp_memory_store):
        store = temp_memory_store
        store.add_note("Henry has a game Saturday", category="sports")
        notes = store.get("notes")
        assert len(notes) == 1
        assert notes[0]["text"] == "Henry has a game Saturday"
        assert notes[0]["category"] == "sports"

    def test_add_rain_out(self, temp_memory_store):
        store = temp_memory_store
        store.add_rain_out("Soccer Practice", "2026-04-20")
        history = store.get("rain_out_history")
        assert len(history) == 1
        assert history[0]["event"] == "Soccer Practice"

    def test_rain_out_history_capped_at_50(self, temp_memory_store):
        store = temp_memory_store
        for i in range(60):
            store.add_rain_out(f"Event {i}", "2026-04-20")
        assert len(store.get("rain_out_history")) <= 50

    def test_add_conversation_summary(self, temp_memory_store):
        store = temp_memory_store
        store.add_conversation_summary("User asked about Henry's schedule.")
        summaries = store.get("conversation_summaries")
        assert len(summaries) == 1

    def test_conversation_summaries_capped_at_20(self, temp_memory_store):
        store = temp_memory_store
        for i in range(25):
            store.add_conversation_summary(f"Summary {i}")
        assert len(store.get("conversation_summaries")) <= 20

    def test_persistence_across_instances(self, tmp_path):
        memory_file = tmp_path / "persist_test.json"
        store1 = MemoryStore(memory_file=memory_file)
        store1.set("persistent_key", "hello")

        store2 = MemoryStore(memory_file=memory_file)
        assert store2.get("persistent_key") == "hello"

    def test_family_context_string(self, temp_memory_store):
        store = temp_memory_store
        context = store.get_family_context_string()
        assert isinstance(context, str)
        assert "Family members" in context
