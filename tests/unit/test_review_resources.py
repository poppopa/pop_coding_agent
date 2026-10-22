"""用户明确派发的审查异常路径I04-I08：日志、脱敏、历史、终端取消。

真实入口保持不变；只替换不可控文件失败/终端输入，并清理本测试进程。
"""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
from io import StringIO
from pathlib import Path
import signal
from types import SimpleNamespace
import sys

import pytest
from rich.console import Console

from coding_agent.domain import Event, ModelFrame
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.storage.session import JsonlSessionStore
from coding_agent.tools.command import RunCommandTool
from coding_agent.ui import terminal


async def test_output_initialization_failure_cannot_spawn_unmanaged_process(file_context, monkeypatch):
    output_directory = file_context.state_dir / "outputs"
    output_directory.parent.mkdir(exist_ok=True)
    output_directory.write_text("blocking-file")
    spawned = []
    original_spawn = asyncio.create_subprocess_exec

    async def spawn_spy(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_spy)
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    file_context.policy.approve_once(argv, file_context.root)
    try:
        result = await RunCommandTool().execute({"argv": argv}, file_context, asyncio.Event())
        assert result.ok is False
        assert result.error is not None
        assert spawned == [], "日志无法准备时不得启动进程"
    finally:
        for process in spawned:
            if process.returncode is None:
                try:
                    import os
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()


async def test_second_secret_occurrence_crossing_publish_cut_is_fully_redacted(file_context):
    secret = "<your-test-key>"
    overlap = len(secret) + 128
    cut = 4096 - overlap
    before_second = secret + "X" * (cut - 3 - len(secret))
    payload = before_second + secret
    payload += "Y" * (4096 - len(payload))
    events = []

    async def emit(kind, data):
        events.append(data)

    file_context.emit = emit
    argv = [sys.executable, "-c", "import os; os.write(1," + repr(payload.encode()) + ")"]
    file_context.policy.approve_once(argv, file_context.root)
    result = await RunCommandTool().execute({"argv": argv}, file_context, asyncio.Event())
    assert result.ok
    visible = "".join(e["text"] for e in events)
    assert secret not in visible
    assert secret not in result.data["stdout"]
    assert visible.count("<REDACTED>") == 2


async def test_history_after_compaction_does_not_mutate_committed_event(tmp_path):
    store = JsonlSessionStore(tmp_path / "state", tmp_path, "reviewhistory")
    original = [{"kind": "message", "role": "user", "content": "压缩前摘要"}]
    try:
        await store.append(Event(1, store.session_id, "t", 1, "context_compacted", {"history": original}))
        await store.append(Event(1, store.session_id, "t", 2, "user_message", {"text": "后续唯一任务"}))
        event_before = deepcopy(store.events[0])
        first = store.history()
        second = store.history()
        assert first == second
        assert store.events[0] == event_before
        assert first == [*original, {"kind": "message", "role": "user", "content": "后续唯一任务"}]
        assert store.events[0].payload["history"] == original
    finally:
        store.close()


def engine_for_terminal(tmp_path):
    return SimpleNamespace(store=SimpleNamespace(directory=tmp_path), context=SimpleNamespace(policy=Policy(), redactor=Redactor()), session_id="review-terminal")


async def test_approval_prompt_does_not_replace_outer_sigint_handler(tmp_path, monkeypatch):
    prompts = iter(["执行命令", "/exit"])
    approval_options = []

    class Prompt:
        def __init__(self, **kwargs):
            self.primary = bool(kwargs)

        async def prompt_async(self, text, **kwargs):
            if self.primary:
                return next(prompts)
            approval_options.append(kwargs)
            return "n"

    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    engine = engine_for_terminal(tmp_path)

    async def run_turn(prompt, cancel):
        assert await engine.approval(["python", "-V"], tmp_path) == "deny"
        yield Event(1, "s", "t", 1, "turn_finished", {"status": "completed"})

    engine.run_turn = run_turn
    assert await terminal.interactive(engine, SimpleNamespace(), Console(file=StringIO())) == 0
    assert approval_options[0].get("handle_sigint") is False


async def test_manual_compact_receives_sigint_and_returns_to_input(tmp_path, monkeypatch):
    prompts = iter(["/compact", "/exit"])

    class Prompt:
        def __init__(self, **kwargs):
            pass

        async def prompt_async(self, text, **kwargs):
            return next(prompts)

    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    engine = engine_for_terminal(tmp_path)
    observed = []
    loop = asyncio.get_running_loop()

    async def compact(cancel):
        loop.call_soon(signal.raise_signal, signal.SIGINT)
        try:
            await asyncio.wait_for(cancel.wait(), 0.3)
            observed.append(cancel.is_set())
            raise asyncio.CancelledError
        except asyncio.TimeoutError:
            observed.append(False)
            return False

    engine.compact = compact
    # 外层测试自己的处理器防止未接管SIGINT打断pytest本身，退出后恢复。
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_args: None)
    try:
        assert await terminal.interactive(engine, SimpleNamespace(), Console(file=StringIO())) == 0
        assert observed == [True]
    finally:
        signal.signal(signal.SIGINT, previous)
