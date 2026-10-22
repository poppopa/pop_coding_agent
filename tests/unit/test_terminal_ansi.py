"""终端恢复契约：真实stdout代理及内核事件，核验原始ANSI和输出清理。"""
import asyncio
from contextlib import contextmanager
from io import StringIO
import sys
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.patch_stdout import StdoutProxy, patch_stdout
from rich.console import Console

from coding_agent import cli
from coding_agent.config import AgentConfig
from coding_agent.domain import Event
from coding_agent.engine import AgentEngine
from coding_agent.providers.ark import ArkProvider
from coding_agent.storage.session import JsonlSessionStore
from coding_agent.ui import terminal
from coding_agent.ui.renderer import JsonRenderer, TerminalRenderer
from test_summary_recovery import Client, call, end, message, seed


class RecordingOutput(DummyOutput):
    def __init__(self):
        self.normal, self.raw = [], []
        self.flushes = 0

    def write(self, text):
        # 模拟终端Output安全文字通道的转义处理，可区分错误的非raw代理。
        self.normal.append(text.replace('\x1b', '?'))

    def write_raw(self, text):
        self.raw.append(text)

    def flush(self):
        self.flushes += 1


@pytest.mark.parametrize('raw', [False, True], ids=['escaped-control', 'raw-preserves-escape'])
def test_real_stdout_proxy_selects_raw_channel_and_closes_flush_thread(raw):
    output = RecordingOutput()
    before = sys.stdout
    with create_app_session(input=DummyInput(), output=output):
        with patch_stdout(raw=raw):
            proxy = sys.stdout
            assert isinstance(proxy, StdoutProxy)
            print('\x1b[36m工具\x1b[0m', flush=True)
        assert sys.stdout is before
        assert proxy.closed
        # close等待队列刷新，不能依赖后台线程碰巧先执行。
        assert not proxy._flush_thread.is_alive()
    assert output.flushes >= 1
    if raw:
        assert output.normal == []
        assert ''.join(output.raw) == '\x1b[36m工具\x1b[0m\n'
    else:
        assert output.raw == []
        assert ''.join(output.normal) == '?[36m工具?[0m\n'


async def test_interactive_uses_real_raw_proxy_for_input_execution_and_approval(file_context, monkeypatch, capsys):
    # SDK替身/输入可控；真实Ark、引擎、六工具、权限及SessionStore都参与。
    config = AgentConfig(model='test-model', state_dir=file_context.state_dir)
    store = JsonlSessionStore(config.state_dir, file_context.root)
    await seed(store, [{'kind': 'message', 'role': 'user', 'content': '只解释原文'},
                      {'kind': 'response', 'items': [message('旧文件正文' * 9000)]},
                      {'kind': 'response', 'items': [message('最近一轮')]},
                      {'kind': 'response', 'items': [message('最近二轮')]}])
    before_log = store.path.read_bytes()
    client = Client([[end(output=[call('read_file', 'read-ansi', path='missing.py'),
                                call('run_command', 'denied-command', argv=['python', '-V'])])],
                     [{'type': 'response.output_text.delta', 'delta': '[red]模型原文[/red]'}, end(output=[message('[red]模型原文[/red]')])]])
    provider = ArkProvider(config=config, client=client)
    engine = AgentEngine(provider, cli.toolset(), store, file_context, config)
    inputs = iter(['继续解释', '/exit'])
    proxies, kwargs_seen, approvals = [], [], []
    output = RecordingOutput()
    original_stdout = sys.stdout

    class Prompt:
        def __init__(self, **kwargs):
            self.primary = bool(kwargs)

        async def prompt_async(self, text, **kwargs):
            assert isinstance(sys.stdout, StdoutProxy) and sys.stdout.raw is True
            if self.primary:
                return next(inputs)
            approvals.append(kwargs)
            return 'n'

    @contextmanager
    def tracked_proxy(**kwargs):
        kwargs_seen.append(kwargs)
        assert kwargs == {'raw': True}
        with patch_stdout(**kwargs):
            proxies.append(sys.stdout)
            yield

    monkeypatch.setattr(terminal, 'PromptSession', Prompt)
    monkeypatch.setattr(terminal, 'patch_stdout', tracked_proxy)
    try:
        with create_app_session(input=DummyInput(), output=output):
            assert await terminal.interactive(engine, config, Console(force_terminal=True, color_system='standard', no_color=False, width=160)) == 0
        assert sys.stdout is original_stdout
        assert len(kwargs_seen) == 4  # 首次输入、执行、审批、退出输入各一次。
        assert approvals == [{'handle_sigint': False}]
        assert all(proxy.closed and not proxy._flush_thread.is_alive() for proxy in proxies)
        rendered = ''.join(output.raw)
        assert '\x1b[36m' in rendered and '\x1b[33m' in rendered
        assert '?[36m' not in rendered and '?[33m' not in rendered
        assert '[red]模型原文[/red]' in rendered
        assert '本地执行事实整理' in rendered and '继续任务' in rendered
        assert output.normal == []
        assert store.path.read_bytes().startswith(before_log)
        assert store.events[-1].payload['status'] == 'completed'
        assert not any(event.type == 'tool_started' and event.payload['call_id'] == 'denied-command' for event in store.events)
        denied = next(event for event in store.events if event.type == 'tool_finished' and event.payload['call_id'] == 'denied-command')
        assert denied.payload['result']['error']['code'] == 'PERMISSION_DENIED'
        assert next(event for event in store.events if event.type == 'context_compacted').payload['source'] == 'local'
        # 命令正文也维持字面渲染；JSONL渲染可解析且没有装饰ANSI。
        command_output = StringIO()
        TerminalRenderer(Console(file=command_output, force_terminal=False)).handle(Event(1, 's', 't', 1, 'tool_output', {'text': '[green]命令原文[/green]'}))
        assert command_output.getvalue() == '[green]命令原文[/green]'
        JsonRenderer().handle(Event(1, 's', 't', 1, 'state_changed', {'state': 'context_ready', 'source': 'local'}))
        import json
        captured = capsys.readouterr().out
        row = json.loads(captured.splitlines()[-1])
        assert row['payload']['source'] == 'local'
        assert '\x1b' not in captured.splitlines()[-1]
    finally:
        store.close()
        await provider.close()


@pytest.mark.parametrize('source', ['local', 'model'])
def test_context_ready_renderer_distinguishes_fact_fallback_from_model(source):
    out = StringIO()
    renderer = TerminalRenderer(Console(file=out, force_terminal=False, width=160))
    renderer.handle(Event(1, 's', 't', 1, 'text_delta', {'text': '前段文字'}))
    renderer.handle(Event(1, 's', 't', 2, 'state_changed', {'state': 'context_ready', 'source': source}))
    text = out.getvalue()
    assert text.startswith('前段文字\n') and '继续任务' in text
    assert ('本地执行事实' in text) is (source == 'local')
    assert '\x1b' not in text


async def test_manual_compact_reports_local_fallback_and_returns_to_input(file_context, monkeypatch):
    config = AgentConfig(model='test-model', state_dir=file_context.state_dir)
    store = JsonlSessionStore(config.state_dir, file_context.root)
    await seed(store, [{'kind': 'message', 'role': 'user', 'content': '当前原文'},
                      *[{'kind': 'response', 'items': [message('完整轮次')]} for _ in range(3)]])
    client = Client([[end('incomplete', reason='max_output_tokens')]])
    provider = ArkProvider(config=config, client=client)
    engine = AgentEngine(provider, cli.toolset(), store, file_context, config)
    prompts = iter(['/compact', '/exit'])
    raw_values = []

    class Prompt:
        def __init__(self, **kwargs):
            pass

        async def prompt_async(self, text):
            return next(prompts)

    @contextmanager
    def proxy(**kwargs):
        raw_values.append(kwargs)
        assert kwargs == {'raw': True}
        yield

    monkeypatch.setattr(terminal, 'PromptSession', Prompt)
    monkeypatch.setattr(terminal, 'patch_stdout', proxy)
    out = StringIO()
    try:
        assert await terminal.interactive(engine, config, Console(file=out, width=160)) == 0
        assert '本地执行事实整理' in out.getvalue() and '完整日志保留' in out.getvalue()
        assert raw_values == [{'raw': True}, {'raw': True}]
        assert len(client.requests) == 1 and store.events[-1].payload['source'] == 'local'
    finally:
        store.close()
        await provider.close()
