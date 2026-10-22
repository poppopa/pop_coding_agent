"""验收依据：技术方案§命令执行与授权；只执行临时目录内固定Python脚本。

单项最多父/子两个进程，继承测试启动器4 GiB上限；finally只清理本用例记录的进程组。
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import stat
import sys

import pytest
from pydantic import ValidationError

from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.tools.base import ToolContext
from coding_agent.tools.command import CommandInput, RunCommandTool


@pytest.fixture
def context(tmp_path, monkeypatch):
    for name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "<your-test-key>")
    events = []

    async def emit(kind, payload):
        events.append((kind, payload))

    root = tmp_path / "repo"
    root.mkdir()
    ctx = ToolContext(root, "test-call", None, Policy(), emit, Redactor(["<your-test-key>"]), tmp_path / "state")
    return ctx, events


async def execute(context, script, *, authorize=True, cancel=None, timeout=5, extra=()):
    ctx, events = context
    argv = [sys.executable, "-c", script, *extra]
    if authorize:
        ctx.policy.approve_once(argv, ctx.root)
    async with asyncio.timeout(10):
        return await RunCommandTool().execute({"argv": argv, "timeout_seconds": timeout, "purpose": "verify"}, ctx, cancel or asyncio.Event())


def test_command_defaults_and_timeout_bounds():
    assert CommandInput(argv=["python", "-V"]).timeout_seconds == 120
    assert CommandInput(argv=["python", "-V"], timeout_seconds=600).timeout_seconds == 600


@pytest.mark.parametrize("arguments", [
    pytest.param({"argv": []}, id="empty-argv"),
    pytest.param({"argv": [""]}, id="empty-program"),
    pytest.param({"argv": ["python", "a\x00b"]}, id="nul"),
    pytest.param({"argv": ["python"], "timeout_seconds": 0}, id="zero-timeout"),
    pytest.param({"argv": ["python"], "timeout_seconds": 601}, id="over-timeout"),
    pytest.param({"argv": "python -V"}, id="shell-string"),
])
def test_command_rejects_invalid_input(arguments):
    with pytest.raises(ValidationError):
        CommandInput.model_validate(arguments)


async def test_argv_is_literal_and_cwd_is_repository(context):
    literal = "$(never-run) ; echo unsafe"
    result = await execute(context, "import sys,os,json; print(json.dumps({'arg':sys.argv[1],'cwd':os.getcwd()}))", extra=[literal])
    assert result.ok and result.data["exit_code"] == 0
    assert json.loads(result.data["stdout"]) == {"arg": literal, "cwd": str(context[0].root)}
    assert result.data["purpose"] == "verify"


async def test_unapproved_command_has_zero_file_side_effect(context):
    result = await execute(context, "from pathlib import Path; Path('marker').write_text('bad')", authorize=False)
    assert result.ok is False
    assert result.error.code == "APPROVAL_REQUIRED"
    assert not (context[0].root / "marker").exists()


async def test_model_credentials_are_removed_from_child_environment(context):
    result = await execute(context, "import os,json; print(json.dumps({k:k in os.environ for k in ['ARK_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY']}))")
    assert result.ok
    assert json.loads(result.data["stdout"]) == {"ARK_API_KEY": False, "OPENAI_API_KEY": False, "ANTHROPIC_API_KEY": False}
    assert "<your-test-key>" not in Path(result.data["output_path"]).read_text()


async def test_pre_cancelled_command_is_not_started(context):
    cancel = asyncio.Event()
    cancel.set()
    result = await execute(context, "from pathlib import Path; Path('marker').touch()", cancel=cancel)
    assert result.error.code == "COMMAND_CANCELLED"
    assert not (context[0].root / "marker").exists()


async def test_command_exit_code_is_preserved(context):
    result = await execute(context, "import sys; print('failure',file=sys.stderr); sys.exit(7)")
    assert not result.ok
    assert result.data["exit_code"] == 7
    assert result.error is not None and result.error.code and result.error.message
    assert result.data["stderr"] == "failure\n"


async def test_dual_pipes_are_drained_after_visible_output_limit(context):
    script = "import os\nfor _ in range(300):\n os.write(1,b'O'*4096)\n os.write(2,b'E'*4096)\nos.write(1,b'STDOUT-END')\nos.write(2,b'STDERR-END')\n"
    result = await execute(context, script)
    assert result.ok and result.truncated
    assert result.data["stdout"].endswith("STDOUT-END")
    assert result.data["stderr"].endswith("STDERR-END")
    assert len(result.data["stdout"].encode()) + len(result.data["stderr"].encode()) <= 32 * 1024
    assert Path(result.data["output_path"]).stat().st_size <= 1024 * 1024
    visible = "".join(p["text"] for kind, p in context[1] if kind == "tool_output")
    assert len(visible.encode()) <= 1024 * 1024 + 256


async def test_command_record_is_private(context):
    result = await execute(context, "print('ok')")
    assert stat.S_IMODE(Path(result.data["output_path"]).stat().st_mode) == 0o600


async def test_output_record_byte_limit_includes_stream_labels(context):
    script = "import os\nfor _ in range(270):\n os.write(1,b'X'*4096)\nos.write(1,b'LAST-BYTES')\n"
    result = await execute(context, script)
    assert result.ok and result.data["stdout"].endswith("LAST-BYTES")
    assert Path(result.data["output_path"]).stat().st_size <= 1024 * 1024


async def test_visible_output_is_bounded_while_tail_still_advances(context):
    result = await execute(context, "import os\nfor _ in range(270):\n os.write(1,b'X'*4096)\nos.write(1,b'LAST-BYTES')\n")
    assert result.ok and result.data["stdout"].endswith("LAST-BYTES")
    visible = "".join(p["text"] for kind, p in context[1] if kind == "tool_output")
    assert len(visible.encode()) <= 1024 * 1024 + 256


async def test_known_placeholder_split_across_chunks_is_redacted_everywhere(context):
    script = "import os,time\nos.write(1,b'X'*4090+b'<your-')\ntime.sleep(.05)\nos.write(1,b'test-key>'+b'Y'*5000)\n"
    result = await execute(context, script)
    assert result.ok
    visible = "".join(p["text"] for kind, p in context[1] if kind == "tool_output")
    record = Path(result.data["output_path"]).read_text()
    for text in (visible, record, result.data["stdout"]):
        assert "<your-test-key>" not in text
        assert "<REDACTED>" in text


async def wait_file(path):
    async with asyncio.timeout(3):
        while not path.exists():
            await asyncio.sleep(0.01)
    return json.loads(path.read_text())


def running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state not in ("Z", "X")
    except FileNotFoundError:
        return False


def cleanup(pids):
    if not pids:
        return
    try:
        os.killpg(pids[0], signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("mode", ["cancel", "timeout", "parent-exit"])
async def test_entire_process_group_is_terminated(context, mode):
    ctx, _ = context
    pid_file = ctx.root / "pids.json"
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    script = (
        "import os,subprocess,sys,json,time,signal\n"
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "open('pids.json','w').write(json.dumps([os.getpid(),child.pid]))\n"
        + ("sys.exit(0)\n" if mode == "parent-exit" else "signal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n")
    )
    cancel = asyncio.Event()
    task = asyncio.create_task(execute(context, script, cancel=cancel, timeout=1 if mode == "timeout" else 5))
    pids = []
    try:
        pids = await wait_file(pid_file)
        if mode == "cancel":
            cancel.set()
        result = await task
        if mode == "cancel":
            assert result.error.code == "COMMAND_CANCELLED"
            assert result.data["cancelled"] is True
        elif mode == "timeout":
            assert result.error.code == "COMMAND_TIMEOUT"
            assert result.data["timed_out"] is True
        assert not running(pids[0]), "父进程仍在运行"
        assert not running(pids[1]), "父进程结束后子进程仍在运行"
    finally:
        cleanup(pids)
        if not task.done():
            cancel.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
