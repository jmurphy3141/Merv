"""Unit tests for the coding agent tools and skill."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from merv.agents.coding_agent import _make_tools, _resolve_safe, run_coding_task
from merv.agents.permissions import Permission, PermissionStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def granted_store(tmp_path: Path) -> PermissionStore:
    """PermissionStore with all permissions pre-granted (for tool tests)."""
    store = PermissionStore(tmp_path / "perms.json")
    for perm in Permission:
        store.grant(perm)
    return store


@pytest.fixture
def empty_store(tmp_path: Path) -> PermissionStore:
    """PermissionStore with no permissions granted."""
    return PermissionStore(tmp_path / "empty_perms.json")


@pytest.fixture
def tools(work_dir: Path, granted_store: PermissionStore):
    return _make_tools(work_dir, granted_store)


@pytest.fixture
def write_file_tool(tools):
    return tools[0]


@pytest.fixture
def read_file_tool(tools):
    return tools[1]


@pytest.fixture
def run_python_tool(tools):
    return tools[2]


@pytest.fixture
def run_tests_tool(tools):
    return tools[3]


# ── _resolve_safe ─────────────────────────────────────────────────────────────

class TestResolveSafe:
    def test_valid_path_returns_path(self, work_dir):
        result = _resolve_safe("foo.py", work_dir)
        assert isinstance(result, Path)
        assert result == (work_dir / "foo.py").resolve()

    def test_parent_traversal_returns_error(self, work_dir):
        result = _resolve_safe("../secret", work_dir)
        assert isinstance(result, str)
        assert "Error" in result

    def test_absolute_outside_returns_error(self, work_dir):
        result = _resolve_safe("/etc/passwd", work_dir)
        assert isinstance(result, str)
        assert "Error" in result

    def test_nested_valid_path(self, work_dir):
        result = _resolve_safe("src/utils.py", work_dir)
        assert isinstance(result, Path)


# ── Permission blocking ───────────────────────────────────────────────────────

class TestToolPermissionBlocking:
    """Each tool must refuse to operate when its permission is not granted."""

    def test_write_file_blocked_without_permission(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        result = tools[0].invoke({"path": "out.py", "content": "x=1"})
        assert "PERMISSION DENIED" in result
        assert "write_files" in result

    def test_read_file_blocked_without_permission(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        result = tools[1].invoke({"path": "out.py"})
        assert "PERMISSION DENIED" in result
        assert "read_files" in result

    def test_run_python_blocked_without_permission(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        result = tools[2].invoke({"script_path": "run.py"})
        assert "PERMISSION DENIED" in result
        assert "execute_code" in result

    def test_run_tests_blocked_without_permission(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        result = tools[3].invoke({"test_path": "."})
        assert "PERMISSION DENIED" in result
        assert "run_tests" in result

    def test_blocked_message_includes_reason(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        result = tools[0].invoke({"path": "out.py", "content": "x=1"})
        assert "Why this permission is needed" in result

    def test_granting_permission_unblocks_tool(self, work_dir, empty_store):
        tools = _make_tools(work_dir, empty_store)
        # blocked before grant
        assert "PERMISSION DENIED" in tools[0].invoke({"path": "ok.py", "content": "x=1"})
        # grant and try again (same store object, so tools see the update)
        empty_store.grant(Permission.WRITE_FILES)
        result = tools[0].invoke({"path": "ok.py", "content": "x=1"})
        assert "Wrote" in result


# ── write_file tool ───────────────────────────────────────────────────────────

class TestWriteFileTool:
    def test_creates_file(self, write_file_tool, work_dir):
        result = write_file_tool.invoke({"path": "hello.py", "content": "print('hi')"})
        assert "Wrote" in result
        assert (work_dir / "hello.py").read_text() == "print('hi')"

    def test_creates_parent_directories(self, write_file_tool, work_dir):
        result = write_file_tool.invoke({"path": "a/b/c.py", "content": "x = 1"})
        assert "Wrote" in result
        assert (work_dir / "a" / "b" / "c.py").exists()

    def test_rejects_path_outside_workdir(self, write_file_tool):
        result = write_file_tool.invoke({"path": "../../evil.py", "content": "bad"})
        assert "Error" in result

    def test_rejects_absolute_outside_path(self, write_file_tool):
        result = write_file_tool.invoke({"path": "/etc/evil", "content": "bad"})
        assert "Error" in result

    def test_reports_line_count(self, write_file_tool):
        content = "a = 1\nb = 2\nc = 3"
        result = write_file_tool.invoke({"path": "lines.py", "content": content})
        assert "3 lines" in result


# ── read_file tool ────────────────────────────────────────────────────────────

class TestReadFileTool:
    def test_reads_existing_file(self, read_file_tool, work_dir):
        work_dir.joinpath("data.txt").write_text("hello world")
        result = read_file_tool.invoke({"path": "data.txt"})
        assert result == "hello world"

    def test_returns_error_for_missing_file(self, read_file_tool):
        result = read_file_tool.invoke({"path": "nope.py"})
        assert "Error" in result
        assert "not found" in result.lower()

    def test_rejects_path_outside_workdir(self, read_file_tool):
        result = read_file_tool.invoke({"path": "../secret"})
        assert "Error" in result

    def test_truncates_large_file(self, read_file_tool, work_dir):
        big = "x" * 15_000
        work_dir.joinpath("big.txt").write_text(big)
        result = read_file_tool.invoke({"path": "big.txt"})
        assert len(result) < 15_000
        assert "truncated" in result


# ── run_python tool ───────────────────────────────────────────────────────────

class TestRunPythonTool:
    def test_runs_hello_world(self, run_python_tool, work_dir):
        work_dir.joinpath("hello.py").write_text("print('hello world')")
        result = run_python_tool.invoke({"script_path": "hello.py"})
        assert "hello world" in result

    def test_captures_stderr(self, run_python_tool, work_dir):
        work_dir.joinpath("err.py").write_text("import sys; sys.stderr.write('oops')")
        result = run_python_tool.invoke({"script_path": "err.py"})
        assert "oops" in result

    def test_reports_exit_code_on_error(self, run_python_tool, work_dir):
        work_dir.joinpath("fail.py").write_text("raise RuntimeError('boom')")
        result = run_python_tool.invoke({"script_path": "fail.py"})
        assert "Exit code" in result
        assert "1" in result

    def test_returns_error_for_missing_script(self, run_python_tool):
        result = run_python_tool.invoke({"script_path": "missing.py"})
        assert "Error" in result

    def test_timeout_returns_error(self, run_python_tool, work_dir):
        work_dir.joinpath("slow.py").write_text("import time; time.sleep(100)")
        with patch("merv.agents.coding_agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python3"], timeout=30)
            result = run_python_tool.invoke({"script_path": "slow.py"})
        assert "timed out" in result.lower()

    def test_rejects_path_outside_workdir(self, run_python_tool):
        result = run_python_tool.invoke({"script_path": "/etc/passwd"})
        assert "Error" in result


# ── run_tests tool ────────────────────────────────────────────────────────────

class TestRunTestsTool:
    def test_passing_test_reports_passed(self, run_tests_tool, work_dir):
        work_dir.joinpath("test_ok.py").write_text(
            "def test_always_passes():\n    assert True\n"
        )
        result = run_tests_tool.invoke({"test_path": "test_ok.py"})
        assert "passed" in result.lower()

    def test_failing_test_reports_failed(self, run_tests_tool, work_dir):
        work_dir.joinpath("test_bad.py").write_text(
            "def test_always_fails():\n    assert False\n"
        )
        result = run_tests_tool.invoke({"test_path": "test_bad.py"})
        assert "failed" in result.lower()

    def test_timeout_returns_error(self, run_tests_tool, work_dir):
        work_dir.joinpath("test_dummy.py").write_text("def test_x(): pass")
        with patch("merv.agents.coding_agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python3"], timeout=30)
            result = run_tests_tool.invoke({"test_path": "test_dummy.py"})
        assert "timed out" in result.lower()

    def test_rejects_path_outside_workdir(self, run_tests_tool):
        result = run_tests_tool.invoke({"test_path": "../../outside"})
        assert "Error" in result


# ── Directory safety (parametrized) ──────────────────────────────────────────

ESCAPE_PATHS = [
    "../secret",
    "/etc/passwd",
    "../../home/user/.ssh/id_rsa",
    "subdir/../../outside",
]


@pytest.mark.parametrize("bad_path", ESCAPE_PATHS)
class TestToolDirectorySafety:
    def test_write_file_rejects(self, bad_path, tools):
        result = tools[0].invoke({"path": bad_path, "content": "bad"})
        assert "Error" in result

    def test_read_file_rejects(self, bad_path, tools):
        result = tools[1].invoke({"path": bad_path})
        assert "Error" in result

    def test_run_python_rejects(self, bad_path, tools):
        result = tools[2].invoke({"script_path": bad_path})
        assert "Error" in result

    def test_run_tests_rejects(self, bad_path, tools):
        result = tools[3].invoke({"test_path": bad_path})
        assert "Error" in result


# ── CodingSkill ───────────────────────────────────────────────────────────────

class TestCodingSkill:
    def _make_state(self, text: str):
        from merv.core.state import MervState
        state = MervState()
        state.add_user_message(text)
        return state

    def _make_mock_settings(self, api_key: str = "sk-test"):
        mock = MagicMock()
        mock.anthropic_api_key = api_key
        mock.llm_model = "claude-sonnet-4-6"
        mock.llm_max_tokens = 1024
        mock.data_dir = None  # overridden by _store_path
        return mock

    def test_no_api_key_returns_friendly_message(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill
        import asyncio

        mock_settings = self._make_mock_settings(api_key="")
        with patch("merv.skills.coding_skill.get_settings", return_value=mock_settings):
            skill = CodingSkill(_store_path=tmp_path / "perms.json")
            result = asyncio.get_event_loop().run_until_complete(
                skill.run(self._make_state("write some code"))
            )
        assert "API key" in result

    async def test_first_task_returns_permission_request(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        mock_settings = self._make_mock_settings()
        with patch("merv.skills.coding_skill.get_settings", return_value=mock_settings):
            skill = CodingSkill(_store_path=tmp_path / "perms.json")
            result = await skill.run(self._make_state("write a fibonacci function"))

        assert "approval" in result.lower() or "permission" in result.lower()
        assert "Why" in result

    async def test_first_task_stores_pending_task(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        mock_settings = self._make_mock_settings()
        store_path = tmp_path / "perms.json"
        with patch("merv.skills.coding_skill.get_settings", return_value=mock_settings):
            skill = CodingSkill(_store_path=store_path)
            await skill.run(self._make_state("write a fibonacci function"))

        store = PermissionStore(store_path)
        assert store.get_pending_task() == "write a fibonacci function"

    async def test_allow_grants_permissions_and_runs_pending(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        store_path = tmp_path / "perms.json"
        # Pre-stage a pending task with no permissions granted
        store = PermissionStore(store_path)
        store.set_pending_task("write a hello world script")

        mock_settings = self._make_mock_settings()
        with (
            patch("merv.skills.coding_skill.get_settings", return_value=mock_settings),
            patch("merv.skills.coding_skill.run_coding_task", new=AsyncMock(return_value="Done!")) as mock_task,
        ):
            skill = CodingSkill(_store_path=store_path)
            result = await skill.run(self._make_state("allow"))

        mock_task.assert_called_once()
        assert "Done!" in result
        # Permissions should be stored
        reloaded = PermissionStore(store_path)
        assert reloaded.missing_permissions() == []

    async def test_deny_stores_denial(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        store_path = tmp_path / "perms.json"
        store = PermissionStore(store_path)
        store.set_pending_task("some task")

        mock_settings = self._make_mock_settings()
        with patch("merv.skills.coding_skill.get_settings", return_value=mock_settings):
            skill = CodingSkill(_store_path=store_path)
            result = await skill.run(self._make_state("deny"))

        assert "denied" in result.lower() or "Denied" in result
        # Pending task should be cleared
        reloaded = PermissionStore(store_path)
        assert reloaded.get_pending_task() is None

    async def test_reset_clears_permissions(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        store_path = tmp_path / "perms.json"
        store = PermissionStore(store_path)
        for perm in Permission:
            store.grant(perm)

        mock_settings = self._make_mock_settings()
        with patch("merv.skills.coding_skill.get_settings", return_value=mock_settings):
            skill = CodingSkill(_store_path=store_path)
            result = await skill.run(self._make_state("reset coding permissions"))

        assert "reset" in result.lower()
        reloaded = PermissionStore(store_path)
        assert reloaded.missing_permissions() == list(Permission)

    async def test_all_granted_calls_run_coding_task(self, tmp_path):
        from merv.skills.coding_skill import CodingSkill

        store_path = tmp_path / "perms.json"
        store = PermissionStore(store_path)
        for perm in Permission:
            store.grant(perm)

        mock_settings = self._make_mock_settings()
        with (
            patch("merv.skills.coding_skill.get_settings", return_value=mock_settings),
            patch("merv.skills.coding_skill.run_coding_task", new=AsyncMock(return_value="Result!")) as mock_task,
        ):
            skill = CodingSkill(_store_path=store_path)
            result = await skill.run(self._make_state("write a sorting function"))

        mock_task.assert_called_once()
        assert mock_task.call_args.kwargs["task"] == "write a sorting function"
        assert result == "Result!"


# ── run_coding_task: recursion limit handling ─────────────────────────────────

class TestRunCodingTask:
    async def test_recursion_error_returns_helpful_message(self, tmp_path):
        from langgraph.errors import GraphRecursionError

        mock_llm = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=GraphRecursionError("limit"))
        store = PermissionStore(tmp_path / "p.json")

        with patch("merv.agents.coding_agent.create_react_agent", return_value=mock_agent):
            result = await run_coding_task("do something", llm=mock_llm, permission_store=store)

        assert "maximum" in result.lower()
        assert "attempts" in result.lower()

    async def test_empty_messages_returns_fallback(self, tmp_path):
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
        store = PermissionStore(tmp_path / "p.json")

        with patch("merv.agents.coding_agent.create_react_agent", return_value=mock_agent):
            result = await run_coding_task("do something", llm=mock_llm, permission_store=store)

        assert result == "No response from coding agent."
