"""摘要恢复契约：真实内核、Ark适配与日志，模型只使用本地SDK事件替身。"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from coding_agent import cli
from coding_agent.config import AgentConfig
from coding_agent.context import ContextLimit
from coding_agent.domain import Event, ModelFrame, ModelRequest
from coding_agent.engine import AgentEngine
from coding_agent.providers import ark
from coding_agent.providers.ark import ArkProvider
from coding_agent.storage.session import JsonlSessionStore


class Stream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        try:
            value = next(self.events)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed += 1


class Client:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.streams = []
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        value = self.outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        stream = value if hasattr(value, '__aiter__') else Stream(value)
        self.streams.append(stream)
        return stream

    async def close(self):
        pass


class RecordingArkProvider(ArkProvider):
    """只记录请求，仍由真实适配器处理SDK事件及关闭资源。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorded = []

    async def stream(self, request, cancel_event):
        self.recorded.append(deepcopy(request))
        async for frame in super().stream(request, cancel_event):
            yield frame


def message(text):
    return {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}


def end(status='completed', output=(), reason=None, **extra):
    response = {'status': status, 'output': list(output), **extra}
    if reason:
        response['incomplete_details'] = {'reason': reason}
    return {'type': 'response.' + status, 'response': response}


def call(name, call_id, **arguments):
    return {'type': 'function_call', 'name': name, 'call_id': call_id,
            'arguments': json.dumps(arguments, ensure_ascii=False)}


def history():
    return [
        {'kind': 'message', 'role': 'user', 'content': '前次摘要：只解释项目，不授权新命令'},
        {'kind': 'message', 'role': 'user', 'content': '旧用户目标甲'},
        {'kind': 'response', 'items': [message('旧助手公开说明'), call('read_file', 'old-read', path='old.py')]},
        {'kind': 'tool_result', 'call_id': 'old-read', 'result': {'call_id': 'old-read', 'ok': True,
         'data': {'path': 'old.py', 'text': '旧文件正文不应进入本地事实' * 180, 'sha256': 'ab' * 32,
                  'size_bytes': 9876, 'start_line': 3, 'start_column': 7, 'end_line': 9,
                  'total_lines': 101, 'next_start_line': 9, 'next_start_column': 12}, 'error': None, 'truncated': True}},
        {'kind': 'response', 'items': [call('apply_patch', 'old-patch', path='changed.py'),
                                      call('run_command', 'old-command', argv=['python', '-V'], cwd='pkg', purpose='verify'),
                                      call('search_text', 'old-search', query='needle', pattern='*.py')]},
        {'kind': 'tool_result', 'call_id': 'old-patch', 'result': {'call_id': 'old-patch', 'ok': True,
         'data': {'path': 'changed.py', 'operation': 'update', 'changed': True, 'sha256': 'cd' * 32}, 'error': None}},
        {'kind': 'tool_result', 'call_id': 'old-command', 'result': {'call_id': 'old-command', 'ok': False,
         'data': {'exit_code': 23, 'purpose': 'verify', 'cancelled': True, 'timed_out': False,
                  'text': '旧命令正文省略'}, 'error': {'code': 'COMMAND_CANCELLED', 'message': '已取消', 'retryable': False}}},
        {'kind': 'tool_result', 'call_id': 'old-search', 'result': {'call_id': 'old-search', 'ok': True,
         'data': {'count': 2, 'skipped_files': 1}, 'error': None}},
        {'kind': 'response', 'items': [call('read_file', 'recent-one', path='recent1.py')]},
        {'kind': 'tool_result', 'call_id': 'recent-one', 'result': {'call_id': 'recent-one', 'ok': True,
         'data': {'text': '最近第一轮原文', 'sha256': 'ef' * 32}, 'error': None}},
        {'kind': 'response', 'items': [message('最近第二轮原文')]},
        {'kind': 'message', 'role': 'user', 'content': '当前目标原文，不能自动执行命令'},
    ]


async def seed(store, entries):
    # 通过真实追加接口提交历史，而非替换恢复函数。
    for entry in entries:
        kind = entry['kind']
        event_type = {'message': 'user_message', 'response': 'model_response', 'tool_result': 'tool_finished'}[kind]
        payload = ({'text': entry['content']} if kind == 'message' else
                   {'items': entry['items']} if kind == 'response' else
                   {'name': 'stored-tool', 'call_id': entry['call_id'], 'result': entry['result']})
        await store.append(Event(1, store.session_id, 'old-turn', store.seq + 1, event_type, payload))


async def build(file_context, outcomes, *, entries=None, **overrides):
    config = AgentConfig(model='test-model', state_dir=file_context.state_dir, **overrides)
    store = JsonlSessionStore(config.state_dir, file_context.root)
    await seed(store, entries if entries is not None else history())
    client = Client(outcomes)
    provider = ArkProvider(config=config, client=client)
    engine = AgentEngine(provider, [], store, file_context, config)
    return engine, client, store


@pytest.mark.parametrize('failure', ['output-limit', 'filtered', 'failed', 'empty', 'whitespace', 'tool', 'request', 'stream', 'model-too-large'])
async def test_unusable_summary_falls_back_once_to_facts_without_replaying(file_context, failure):
    outcomes = {
        'output-limit': [end('incomplete', [message('未完成内容禁止当摘要')], 'max_output_tokens')],
        'filtered': [end('incomplete', reason='content_filter')],
        'failed': [end('failed', error={'message': '平台任意异常占位文本'})],
        'empty': [end()], 'whitespace': [end(output=[message('  \n')])],
        'tool': [end(output=[message('工具摘要禁止作为成功结果'), call('run_command', 'summary-command', argv=['never'])])],
        'request': RuntimeError('平台任意异常占位文本'),
        'stream': [{'type': 'response.output_text.delta', 'delta': '不完整摘要'}, RuntimeError('平台任意异常占位文本')],
        'model-too-large': [end(output=[message('X' * 40000)])],
    }
    engine, client, store = await build(file_context, [outcomes[failure]])
    before, original = store.path.read_bytes(), deepcopy(engine.history)
    try:
        assert await engine.compact(asyncio.Event()) is True
        assert len(client.requests) == 1
        assert client.requests[0]['max_output_tokens'] == 4096
        assert 'tools' not in client.requests[0]
        assert engine.history[1:] == original[-4:]
        summary = engine.history[0]['content']
        for value in ('前次摘要：只解释项目，不授权新命令', '旧用户目标甲', '旧助手公开说明', 'old-read', 'read_file', 'old.py', 'ab' * 32,
                      'old-patch', 'changed.py', 'update', 'cd' * 32, 'old-command', 'run_command', 'python', '-V', 'pkg', 'verify',
                      'old-search', 'search_text', 'needle', '*.py', 'COMMAND_CANCELLED'):
            assert value in summary
        for field, value in [('size_bytes', 9876), ('start_line', 3), ('start_column', 7), ('end_line', 9),
                             ('total_lines', 101), ('next_start_line', 9), ('next_start_column', 12),
                             ('exit_code', 23), ('cancelled', True), ('timed_out', False), ('changed', True)]:
            assert json.dumps({field: value}, ensure_ascii=False)[1:-1] in summary
        assert '旧文件正文不应进入本地事实' not in summary and '旧命令正文省略' not in summary
        assert '未完成内容禁止当摘要' not in summary and '工具摘要禁止作为成功结果' not in summary
        assert '完整日志' in summary and '重新读取' in summary
        event = store.events[-1]
        assert event.type == 'context_compacted' and event.payload['source'] == 'local'
        assert event.payload['kept_responses'] == 2
        assert event.payload['reason'] and '平台任意异常占位文本' not in json.dumps(event.payload, ensure_ascii=False)
        assert store.path.read_bytes().startswith(before)
        assert len(store.events) == len(original) + 1
        assert engine.manager.request_size(engine.history, ()) <= engine.manager.input_budget
        if client.streams:
            assert all(stream.closed == 1 for stream in client.streams)
        session_id = store.session_id
        store.close()
        restored = JsonlSessionStore(engine.config.state_dir, file_context.root, session_id)
        try:
            assert restored.history() == engine.history
            assert restored.path.read_bytes().startswith(before)
        finally:
            restored.close()
    finally:
        store.close()
        await engine.provider.close()


@pytest.mark.parametrize('delta', [False, True], ids=['terminal-only', 'delta-no-duplication'])
async def test_complete_summary_uses_terminal_text_or_delta_and_config_budget(file_context, delta):
    events = ([{'type': 'response.output_text.delta', 'delta': '增量摘要'}] if delta else []) + [end(output=[message('完整終态甲'), message('完整终态乙')])]
    engine, client, store = await build(file_context, [events], max_output_tokens=2048)
    original = deepcopy(engine.history)
    try:
        assert await engine.compact(asyncio.Event())
        assert engine.history[0]['content'] == '先前任务摘要：\n' + ('增量摘要' if delta else '完整終态甲完整终态乙')
        assert engine.history[1:] == original[-4:]
        assert client.requests[0]['max_output_tokens'] == 2048
        assert store.events[-1].payload['source'] == 'model' and store.events[-1].payload['reason'] == ''
        assert client.streams[0].closed == 1
    finally:
        store.close()
        await engine.provider.close()


@pytest.mark.parametrize('oversized', ['old-body', 'facts', 'recent', 'latest'])
async def test_summary_input_precheck_and_local_hard_budget_preserve_original(file_context, oversized):
    entries = history()
    if oversized == 'old-body':
        entries[3]['result']['data']['text'] = '正文' * 20000
    elif oversized == 'facts':
        entries[4]['items'][1]['arguments'] = json.dumps({'argv': ['X' * 40000], 'purpose': 'verify'})
    elif oversized == 'recent':
        entries[-3]['result']['data']['text'] = '最近完整正文' * 6000
    else:
        entries[-2]['items'] = [message('最新完整正文' * 6000)]
    engine, client, store = await build(file_context, [[end('incomplete', reason='max_output_tokens')]], entries=entries)
    before, original = store.path.read_bytes(), deepcopy(engine.history)
    try:
        if oversized == 'old-body':
            assert await engine.compact(asyncio.Event())
            assert client.requests == []
            assert store.events[-1].payload['source'] == 'local'
            assert engine.manager.request_size(engine.history, ()) <= engine.manager.input_budget
        elif oversized == 'recent':
            # 正式第二轮规格修订：预算不足时完整较早轮移为事实，最新轮不截断。
            assert await engine.compact(asyncio.Event())
            assert engine.history[1:] == original[-2:]
            assert store.events[-1].payload['kept_responses'] == 1
            assert 'recent-one' in engine.history[0]['content']
            assert '最近完整正文' not in engine.history[0]['content']
            assert engine.manager.request_size(engine.history, ()) <= engine.manager.input_budget
            assert store.path.read_bytes().startswith(before)
        else:
            with pytest.raises(ContextLimit, match='可用输入预算'):
                await engine.compact(asyncio.Event())
            assert engine.history == original and store.path.read_bytes() == before
            assert len(client.requests) <= 1
    finally:
        store.close()
        await engine.provider.close()


@pytest.mark.parametrize('adapter', ['ark', 'injected'])
async def test_cancellation_closes_summary_and_never_falls_back(file_context, adapter):
    entered, closed = asyncio.Event(), asyncio.Event()
    engine, client, store = await build(file_context, [])

    class WaitingStream(Stream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            await super().close()
            closed.set()

    async def injected(request, cancel):
        entered.set()
        try:
            await asyncio.Event().wait()
            yield ModelFrame('response_finished', status='completed')
        finally:
            closed.set()

    if adapter == 'ark':
        client.outcomes = [WaitingStream([])]
    else:
        engine.provider = SimpleNamespace(stream=injected)
    before, original = store.path.read_bytes(), deepcopy(engine.history)
    cancel = asyncio.Event()
    task = asyncio.create_task(engine.compact(cancel))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            cancel.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await closed.wait()
        assert engine.history == original and store.path.read_bytes() == before
        assert len(client.requests) == (1 if adapter == 'ark' else 0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        store.close()
        if adapter == 'ark':
            await engine.provider.close()


@pytest.mark.parametrize('reason,diagnostic', [('max_output_tokens', 'PROVIDER_OUTPUT_LIMIT'), ('content_filter', 'PROVIDER_CONTENT_FILTER'), ('other', 'PROVIDER_PROTOCOL_ERROR')])
async def test_real_ark_incomplete_diagnostic_is_fixed_and_preserves_tool_schema(file_context, reason, diagnostic):
    client = Client([[end('incomplete', reason=reason, internal_details='平台任意异常占位文本')]])
    provider = ArkProvider(client=client, model='test-model')
    schema = ({'type': 'function', 'name': 'read_file', 'parameters': {'type': 'object'}},)
    try:
        frames = [frame async for frame in provider.stream(ModelRequest('固定约定', (), schema, 4096), asyncio.Event())]
        assert frames[-1].status == 'incomplete' and frames[-1].text.startswith(diagnostic + ':')
        assert frames[-1].calls == () and '平台任意异常占位文本' not in frames[-1].text
        assert client.requests[0]['tools'] == list(schema)
        assert client.streams[0].closed == 1
    finally:
        await provider.close()


async def test_retryable_summary_request_still_makes_at_most_one_model_attempt(file_context, monkeypatch):
    # 摘要恢复契约明确限制最多一次摘要请求，不能沿用普通响应重试放大尝试。
    monkeypatch.setattr(ark.asyncio, 'sleep', AsyncMock())
    engine, client, store = await build(file_context, [httpx.ConnectError('固定测试连接错误')] * 3)
    try:
        assert await engine.compact(asyncio.Event())
        assert len(client.requests) == 1
        assert store.events[-1].payload['source'] == 'local'
    finally:
        store.close()
        await engine.provider.close()


async def test_real_cli_reads_two_finds_four_then_three_files_and_continues_after_summary_failure(tmp_path, monkeypatch, capsys):
    for name in ('ARK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    root, state = tmp_path / 'repo', tmp_path / 'state'
    root.mkdir()
    for i in range(7):
        (root / f'module{i}.py').write_text(chr(97 + i) * 10000)
    find = [call('find_files', f'find-{i}', pattern='*.py') for i in range(2)]
    reads = [call('read_file', f'read-{i}', path=f'module{i}.py') for i in range(7)]
    client = Client([[end(output=find)], [end(output=reads[:4])], [end(output=reads[4:])],
                     [end('incomplete', [message('失败摘要不能记录为成功')], 'max_output_tokens')],
                     [{'type': 'response.output_text.delta', 'delta': '项目解释正常完成'}, end(output=[message('项目解释正常完成')])]])
    config = AgentConfig(model='test-model', state_dir=state)
    provider = RecordingArkProvider(config=config, client=client)
    prompt = '解释一下当前这个项目，读更多文件后继续说明'
    args = cli.parser().parse_args(['--root', str(root), 'exec', prompt, '--json'])
    assert await cli.run_async(args, config, {'provider': provider}) == 0
    captured = capsys.readouterr()
    assert captured.err == '' and '\x1b' not in captured.out
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert rows[-1]['payload']['status'] == 'completed'
    assert any(row['type'] == 'text_delta' and row['payload']['text'] == '项目解释正常完成' for row in rows)
    directory = next((state / 'sessions').iterdir())
    original_log = directory.joinpath('events.jsonl').read_bytes()
    stored = [json.loads(line) for line in original_log.splitlines()]
    # 内核持久化压缩快照，CLI通过context_ready通知，原始日志仍是恢复证据。
    compacted = [row for row in stored if row['type'] == 'context_compacted']
    assert len(compacted) == 1 and compacted[0]['payload']['source'] == 'local'
    assert compacted[0]['payload']['reason'] == 'summary_output_limit'
    assert compacted[0]['payload']['kept_responses'] == 1
    assert len(client.requests) == 5 and 'tools' not in client.requests[3]
    final_input = client.requests[-1]['input']
    assert any(item.get('content') == prompt for item in final_input)
    assert [item['call_id'] for item in final_input if item.get('type') == 'function_call'] == [item['call_id'] for item in reads[4:]]
    outputs = [item for item in final_input if item.get('type') == 'function_call_output']
    assert [item['call_id'] for item in outputs] == [item['call_id'] for item in reads[4:]]
    assert all(json.loads(item['output'])['data']['text'] == chr(97 + i) * 4096 for i, item in enumerate(outputs, start=4))
    summary = compacted[0]['payload']['history'][0]['content']
    import hashlib
    for i in range(4):
        assert f'module{i}.py' in summary and f'read-{i}' in summary
        assert hashlib.sha256((chr(97 + i) * 10000).encode()).hexdigest() in summary
        assert chr(97 + i) * 4096 not in summary
    for request in provider.recorded:
        if request.tool_schemas:
            used = len(json.dumps([request.instructions, request.history, request.tool_schemas], ensure_ascii=False).encode()) + 512
            assert used <= 25856
    assert [row['payload']['call_id'] for row in rows if row['type'] == 'tool_started'] == [item['call_id'] for item in find + reads]
    assert [row for row in stored if row['type'] != 'context_compacted'] == rows
    assert sum(row['type'] == 'tool_finished' for row in rows) == 9
    for i, row in enumerate([row for row in stored if row['type'] == 'tool_finished' and row['payload']['name'] == 'read_file']):
        assert row['payload']['result']['data']['text'] == chr(97 + i) * 4096
    restored = JsonlSessionStore(state, root, directory.name)
    try:
        assert restored.path.read_bytes() == original_log
        assert restored.history()[-1]['items'] == [message('项目解释正常完成')]
    finally:
        restored.close()


async def test_two_complete_responses_over_hard_budget_compact_older_group(file_context):
    entries = history()
    entries = [entries[0], *entries[2:4], *entries[-4:-2], entries[-1]]
    entries[2]['result']['data']['text'] = '旧文件' * 11000
    engine, client, store = await build(file_context, [], entries=entries)
    before = store.path.read_bytes()
    original = deepcopy(engine.history)
    try:
        assert sum(entry['kind'] == 'response' for entry in original) == 2
        assert await engine.compact(asyncio.Event())
        assert client.requests == []
        assert store.events[-1].payload['kept_responses'] == 1
        assert engine.history[1:] == original[-3:]
        assert engine.manager.request_size(engine.history, ()) <= engine.manager.input_budget
        assert store.path.read_bytes().startswith(before)
    finally:
        store.close()
        await engine.provider.close()


async def test_missing_complete_terminal_uses_fixed_local_reason(file_context):
    engine, client, store = await build(file_context, [])
    closed = asyncio.Event()

    async def stream(request, cancel):
        try:
            yield ModelFrame('text_delta', text='没有完整终态的摘要不得采用')
        finally:
            closed.set()

    engine.provider = SimpleNamespace(stream=stream)
    try:
        assert await engine.compact(asyncio.Event())
        assert closed.is_set()
        assert store.events[-1].payload['source'] == 'local'
        assert store.events[-1].payload['reason'] == 'summary_missing_terminal'
        assert '没有完整终态的摘要不得采用' not in engine.history[0]['content']
    finally:
        store.close()


async def test_model_request_new_flag_defaults_to_existing_retry_behavior(file_context, monkeypatch):
    monkeypatch.setattr(ark.asyncio, 'sleep', AsyncMock())
    request = ModelRequest('固定约定', (), (), 4096)
    assert request.retry_on_connection_error is True
    client = Client([httpx.ConnectError('固定连接错误'), [end()]])
    provider = ArkProvider(client=client, model='test-model')
    try:
        frames = [frame async for frame in provider.stream(request, asyncio.Event())]
        assert frames[-1].status == 'completed' and len(client.requests) == 2
    finally:
        await provider.close()


def summary_facts(entry):
    # 仅解析对外事件中的事实JSON，用于逐字段比较；不调用本地摘要实现生成期望。
    assert entry['summary_source'] == 'local'
    return json.loads(entry['content'].split('\n', 1)[1])


async def test_repeated_local_compaction_after_session_restore_keeps_all_facts_and_user_tail(file_context):
    long_user = '旧用户完整约束' * 240 + '尾部约束：不得扩权，不得自动重放验证命令'
    entries = history()
    entries.insert(1, {'kind': 'message', 'role': 'user', 'content': long_user})
    # 末尾验证事实落在第一次摘要1600字符之后，重复摘要必须保持结构化事实。
    entries[3]['items'].extend([call('run_command', f'tail-command-{i}', argv=['python', '-m', f'verify_{i}'],
                                   cwd=f'pkg_{i}', purpose='verify') for i in range(6)])
    insertion = 5
    for i in range(6):
        entries.insert(insertion + i, {'kind': 'tool_result', 'call_id': f'tail-command-{i}',
            'result': {'call_id': f'tail-command-{i}', 'ok': i % 2 == 0,
                       'error': None if i % 2 == 0 else {'code': 'COMMAND_TIMEOUT', 'message': '验证超时', 'retryable': False},
                       'data': {'exit_code': i, 'purpose': 'verify', 'cancelled': False, 'timed_out': i % 2 == 1,
                                'sha256': f'{i + 1:064x}'}}})
    engine, client, store = await build(file_context, [[end('incomplete', reason='max_output_tokens')]] * 3, entries=entries)
    before_log = store.path.read_bytes()
    provider = engine.provider
    try:
        assert await engine.compact(asyncio.Event())
        first = deepcopy(engine.history[0])
        assert len(first['content']) > 1600
        assert first['content'].find('tail-command-5') > 1600
        first_facts = summary_facts(first)
        assert {'role': 'user', 'content': long_user} in first_facts
        expected_tail = {'tool_call': {'name': 'run_command', 'call_id': 'tail-command-5',
                          'arguments': {'argv': ['python', '-m', 'verify_5'], 'cwd': 'pkg_5', 'purpose': 'verify'}}}
        assert expected_tail in first_facts
        expected_result = {'tool_result': {'call_id': 'tail-command-5', 'name': 'run_command', 'ok': False,
                           'error': {'code': 'COMMAND_TIMEOUT', 'message': '验证超时', 'retryable': False}, 'truncated': False,
                           'data': {'exit_code': 5, 'purpose': 'verify', 'cancelled': False, 'timed_out': True, 'sha256': f'{6:064x}'}}}
        assert expected_result in first_facts
        for cycle in (2, 3):
            session_id = store.session_id
            current = deepcopy(engine.history)
            store.close()
            store = JsonlSessionStore(engine.config.state_dir, file_context.root, session_id)
            assert store.history() == current
            await seed(store, [{'kind': 'response', 'items': [message(f'后续响应甲{cycle}')]},
                               {'kind': 'response', 'items': [message(f'后续响应乙{cycle}')]},
                               {'kind': 'message', 'role': 'user', 'content': f'后续目标{cycle}，不授权重放'}])
            # 模拟后续轮次已经完整提交，再从真实store构造内核，序号随日志恢复。
            engine = AgentEngine(provider, [], store, file_context, engine.config)
            assert await engine.compact(asyncio.Event())
            facts = summary_facts(engine.history[0])
            for fact in first_facts:
                assert facts.count(fact) == first_facts.count(fact)
            assert expected_tail in facts and expected_result in facts
            assert {'role': 'user', 'content': long_user} in facts
            # 旧摘要JSON不能作为用户文字递归嵌套，末尾事实也不能被截掉。
            assert not any(fact.get('content') == first['content'] for fact in facts)
            assert engine.history[-1]['content'] == f'后续目标{cycle}，不授权重放'
            assert [entry['items'] for entry in engine.history if entry['kind'] == 'response'] == [
                [message(f'后续响应甲{cycle}')], [message(f'后续响应乙{cycle}')]]
            assert engine.manager.request_size(engine.history, ()) <= engine.manager.input_budget
        assert len(client.requests) == 3
        assert sum(event.type == 'context_compacted' for event in store.events) == 3
        assert not any(event.type == 'tool_started' for event in store.events)
        assert store.path.read_bytes().startswith(before_log)
    finally:
        store.close()
        await provider.close()


@pytest.mark.parametrize('operation', ['create', 'update', 'delete', 'snapshot-failure'])
async def test_local_summary_preserves_real_patch_fingerprints_existence_and_side_effect_error(file_context, monkeypatch, operation):
    from dataclasses import asdict, replace
    import hashlib
    from coding_agent.tools.patch import ApplyPatchTool
    target = file_context.root / 'actual_patch.py'
    original = b'value = 1\n'
    replacement = b'value = 2\n'
    if operation != 'create':
        target.write_bytes(original)
    if operation == 'create':
        arguments = {'path': target.name, 'operation': 'create', 'content': replacement.decode()}
    elif operation == 'delete':
        arguments = {'path': target.name, 'operation': 'delete', 'expected_sha256': hashlib.sha256(original).hexdigest()}
    else:
        arguments = {'path': target.name, 'operation': 'update', 'expected_sha256': hashlib.sha256(original).hexdigest(),
                     'edits': [{'old_text': '1', 'new_text': '2'}]}
    if operation == 'snapshot-failure':
        def broken_snapshot(path, before, after):
            raise OSError('固定快照写入错误')
        # 只隔离故障磁盘保存，补丁发布、指纹和副作用仍是实际工具行为。
        monkeypatch.setattr(file_context.snapshots, 'record_change', broken_snapshot)
    result = await ApplyPatchTool().execute(arguments, replace(file_context, call_id='actual-patch'), asyncio.Event())
    assert result.ok is (operation != 'snapshot-failure')
    assert result.data['changed'] is True
    if operation == 'snapshot-failure':
        assert result.error.code == 'SNAPSHOT_WRITE_FAILED' and target.read_bytes() == replacement
    entries = [{'kind': 'message', 'role': 'user', 'content': '保持真实修改事实，不把快照失败当回滚'},
               {'kind': 'response', 'items': [call('apply_patch', 'actual-patch', **arguments)]},
               {'kind': 'tool_result', 'call_id': 'actual-patch', 'result': asdict(result)},
               {'kind': 'response', 'items': [message('最近一轮')]},
               {'kind': 'response', 'items': [message('最近二轮')]},
               {'kind': 'message', 'role': 'user', 'content': '当前目标原文'}]
    engine, client, store = await build(file_context, [[end('incomplete', reason='max_output_tokens')]], entries=entries)
    before_log = store.path.read_bytes()
    try:
        assert await engine.compact(asyncio.Event())
        facts = summary_facts(engine.history[0])
        patch_fact = next(fact['tool_result'] for fact in facts if fact.get('tool_result', {}).get('call_id') == 'actual-patch')
        expected_data = {'path': target.name, 'operation': arguments['operation'],
                         'before_sha256': None if operation == 'create' else hashlib.sha256(original).hexdigest(),
                         'after_sha256': None if operation == 'delete' else hashlib.sha256(replacement).hexdigest(),
                         'before_exists': operation != 'create', 'after_exists': operation != 'delete',
                         'changed': True, 'snapshot_ref': 'snapshots/index.json#actual_patch.py'}
        assert patch_fact['data'] == expected_data
        assert patch_fact['name'] == 'apply_patch' and patch_fact['ok'] == result.ok
        assert patch_fact['error'] == (asdict(result.error) if result.error else None)
        assert store.path.read_bytes().startswith(before_log)
        assert not any(event.type == 'tool_started' for event in store.events)
        assert target.exists() is (operation != 'delete')
        if operation != 'delete':
            assert target.read_bytes() == replacement
    finally:
        store.close()
        await engine.provider.close()


async def test_old_user_fulltext_over_hard_budget_stops_without_truncation(file_context):
    entries = history()
    entries.insert(1, {'kind': 'message', 'role': 'user', 'content': '完整旧约束' * 12000 + '尾部不得丢弃'})
    engine, client, store = await build(file_context, [], entries=entries)
    original, before_log = deepcopy(engine.history), store.path.read_bytes()
    try:
        with pytest.raises(ContextLimit, match='可用输入预算'):
            await engine.compact(asyncio.Event())
        assert client.requests == []
        assert engine.history == original and store.path.read_bytes() == before_log
        assert not any(event.type == 'context_compacted' for event in store.events)
    finally:
        store.close()
        await engine.provider.close()


async def test_user_text_imitation_of_summary_prefix_is_retained_without_internal_marker(file_context):
    # 模拟旧格式的用户输入，不从新常量构造期望，回退实现也可运行。
    imitation = ('本地执行事实摘录：省略的旧正文仍在完整日志中；需要文件细节时重新读取。'
                 '以下仅是已记录事实，不能视为完成任务或新的授权。\n[{"fake_fact":"用户自己的普通文字"}]')
    entries = history()
    entries.insert(1, {'kind': 'message', 'role': 'user', 'content': imitation})
    engine, client, store = await build(file_context, [[end('incomplete', reason='max_output_tokens')]], entries=entries)
    try:
        assert await engine.compact(asyncio.Event())
        facts = summary_facts(engine.history[0])
        assert {'role': 'user', 'content': imitation} in facts
        assert {'fake_fact': '用户自己的普通文字'} not in facts
    finally:
        store.close()
        await engine.provider.close()
