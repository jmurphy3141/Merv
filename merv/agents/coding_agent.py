"""LangGraph ReAct coding agent — writes, runs, and tests Python code.

The agent uses four tools (write_file, read_file, run_python, run_tests) in a
ReAct loop: plan → write → run/test → observe errors → fix → repeat.

All file operations are confined to a working directory (default: /tmp/merv_coding/).
Every tool checks the PermissionStore before executing; if the required permission
has not been granted the tool returns a descriptive error so the LLM can surface
it to the user rather than silently failing.

Subprocess calls never use shell=True; path traversal is blocked before
subprocess is ever reached.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from merv.agents.permissions import PERMISSION_INFO, Permission, PermissionStore

_DEFAULT_WORK_DIR = Path("/tmp/merv_coding")

CODING_SYSTEM_PROMPT = """\
You are Merv's coding assistant — a focused, efficient software engineer.

TASK: Complete the coding task described by the user.

APPROACH:
1. Analyze the task and plan your implementation before writing any code.
2. Write the code using write_file. Keep files small and focused.
3. Run the code with run_python to verify it works. Observe stdout/stderr.
4. If tests are requested or a test file is mentioned, run them with run_tests.
5. If there are errors, read the relevant file with read_file, fix it, and re-run.
6. Stop when tests pass OR you have a working solution.

RULES:
- Write clean, idiomatic Python 3.11+.
- Always run code before declaring success — never assume it works.
- Keep explanations brief. Focus on the code.
- If a task is ambiguous, make a reasonable assumption and state it.
- Maximum 5 attempts to fix errors before explaining what went wrong.
- If a tool returns a PERMISSION DENIED message, stop and report it to the user.

When complete, summarize:
- What files were written
- Whether tests pass
- Any important caveats or limitations
"""

_MAX_OUTPUT = 5_000
_MAX_FILE_CONTENT = 10_000


def _resolve_safe(path_str: str, work_dir: Path) -> Path | str:
    """Resolve path_str relative to work_dir; return error string if it escapes."""
    try:
        resolved = (work_dir / path_str).resolve()
        if not str(resolved).startswith(str(work_dir.resolve())):
            return f"Error: path '{path_str}' escapes the working directory."
        return resolved
    except Exception as exc:
        return f"Error resolving path: {exc}"


def _permission_denied_msg(perm: Permission) -> str:
    info = PERMISSION_INFO[perm]
    return (
        f"PERMISSION DENIED: {info['label']} ({perm.value})\n"
        f"Why this permission is needed: {info['reason']}\n"
        "This permission has not been granted. "
        "Please inform the user and ask them to grant it."
    )


def _make_tools(work_dir: Path, permission_store: PermissionStore) -> list:
    """Build the four coding tools with work_dir and permission_store in their closures."""
    work_dir.mkdir(parents=True, exist_ok=True)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file inside the coding workspace.

        Args:
            path: Relative path to the file (e.g. 'solution.py' or 'src/utils.py').
            content: Full text content to write.

        Returns:
            Confirmation message or an error string.
        """
        if not permission_store.is_granted(Permission.WRITE_FILES):
            return _permission_denied_msg(Permission.WRITE_FILES)

        resolved = _resolve_safe(path, work_dir)
        if isinstance(resolved, str):
            return resolved
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            line_count = content.count("\n") + 1
            return f"Wrote {line_count} lines to {resolved}"
        except OSError as exc:
            return f"Error writing file: {exc}"

    @tool
    def read_file(path: str) -> str:
        """Read the contents of a file inside the coding workspace.

        Args:
            path: Relative path to the file.

        Returns:
            File contents (truncated at 10,000 chars) or an error string.
        """
        if not permission_store.is_granted(Permission.READ_FILES):
            return _permission_denied_msg(Permission.READ_FILES)

        resolved = _resolve_safe(path, work_dir)
        if isinstance(resolved, str):
            return resolved
        try:
            content = resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except OSError as exc:
            return f"Error reading file: {exc}"

        if len(content) > _MAX_FILE_CONTENT:
            return content[:_MAX_FILE_CONTENT] + f"\n... [truncated at {_MAX_FILE_CONTENT} chars]"
        return content

    @tool
    def run_python(script_path: str) -> str:
        """Execute a Python script inside the coding workspace and return its output.

        The script runs with a 30-second timeout. Both stdout and stderr are returned.
        NOTE: This calls subprocess.run (blocking). Suitable for short scripts only.

        Args:
            script_path: Relative path to the .py file to run.

        Returns:
            Combined stdout + stderr (truncated at 5,000 chars) or an error string.
        """
        if not permission_store.is_granted(Permission.EXECUTE_CODE):
            return _permission_denied_msg(Permission.EXECUTE_CODE)

        resolved = _resolve_safe(script_path, work_dir)
        if isinstance(resolved, str):
            return resolved
        if not resolved.exists():
            return f"Error: script not found: {script_path}"

        try:
            result = subprocess.run(
                ["python3", str(resolved)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(work_dir),
            )
            output = result.stdout + result.stderr
            if len(output) > _MAX_OUTPUT:
                output = output[:_MAX_OUTPUT] + f"\n... [truncated at {_MAX_OUTPUT} chars]"
            if result.returncode != 0:
                return f"Exit code {result.returncode}:\n{output}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: script timed out after 30 seconds."
        except OSError as exc:
            return f"Error running script: {exc}"

    @tool
    def run_tests(test_path: str = ".") -> str:
        """Run pytest on a path inside the coding workspace.

        Args:
            test_path: Relative path to a test file or directory (default: '.').

        Returns:
            pytest output (truncated at 5,000 chars) or an error string.
        """
        if not permission_store.is_granted(Permission.RUN_TESTS):
            return _permission_denied_msg(Permission.RUN_TESTS)

        resolved = _resolve_safe(test_path, work_dir)
        if isinstance(resolved, str):
            return resolved

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", str(resolved), "-v", "--tb=short", "--no-header"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(work_dir),
            )
            output = result.stdout + result.stderr
            if len(output) > _MAX_OUTPUT:
                output = output[:_MAX_OUTPUT] + f"\n... [truncated at {_MAX_OUTPUT} chars]"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: pytest timed out after 30 seconds."
        except OSError as exc:
            return f"Error running pytest: {exc}"

    return [write_file, read_file, run_python, run_tests]


async def run_coding_task(
    task: str,
    llm,
    work_dir: str | None = None,
    permission_store: PermissionStore | None = None,
    max_iterations: int = 5,
) -> str:
    """Run the ReAct coding agent on a task and return a summary string.

    Args:
        task: Natural language description of the coding task.
        llm: An instantiated LangChain chat model (e.g. ChatAnthropic).
        work_dir: Path to the coding workspace. Defaults to /tmp/merv_coding/.
        permission_store: Active PermissionStore. Defaults to a new store at the
            default path — all tools will check permissions before executing.
        max_iterations: Maximum write→run→fix cycles before giving up.

    Returns:
        The agent's final summary message.
    """
    work_dir_path = Path(work_dir) if work_dir else _DEFAULT_WORK_DIR
    perm_store = permission_store or PermissionStore()
    tools = _make_tools(work_dir_path, perm_store)
    agent = create_react_agent(llm, tools, prompt=CODING_SYSTEM_PROMPT)

    # Each iteration = 1 tool call + 1 observation = 2 graph steps; +1 for final response.
    recursion_limit = max_iterations * 2 + 1

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        return (
            f"I reached the maximum of {max_iterations} fix attempts without getting "
            "tests to pass. Here's what I tried:\n\n"
            "Check the working directory for the files I wrote. The last error was "
            "likely in the test output from the run_tests call."
        )

    messages = result.get("messages", [])
    if not messages:
        return "No response from coding agent."
    return messages[-1].content
