"""无 Key 演示工具循环：真实读文件、应用补丁、运行测试；模型由固定替身提供。"""
from __future__ import annotations
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from coding_agent.config import AgentConfig
from coding_agent.domain import ModelFrame, ToolCall
from coding_agent.engine import AgentEngine
from coding_agent.cli import toolset
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.storage.session import JsonlSessionStore
from coding_agent.storage.snapshots import SnapshotStore
from coding_agent.tools.base import ToolContext
from coding_agent.ui.renderer import TerminalRenderer

class DemoProvider:
    def __init__(self, root):
        self.root = root
        self.step = 0

    async def stream(self, request, cancel_event):
        self.step += 1
        if self.step == 1:
            name, args = 'read_file', {'path': 'calculator.py'}
        elif self.step == 2:
            content = (self.root / 'calculator.py').read_bytes()
            name, args = 'apply_patch', {'path': 'calculator.py', 'operation': 'update',
                'expected_sha256': hashlib.sha256(content).hexdigest(),
                'edits': [{'old_text': '    return sum(values) / len(values)',
                           'new_text': '    # 空集合没有平均值，按契约返回 0。\n    return sum(values) / len(values) if values else 0'}]}
        elif self.step == 3:
            name, args = 'run_command', {'argv': [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts='], 'purpose': 'verify'}
        else:
            yield ModelFrame('text_delta', text='演示完成：空列表分支已修复，验证结果以工具退出状态为准。\n')
            yield ModelFrame('response_finished', status='completed', provider_output=({'type':'message','role':'assistant','content':[{'type':'output_text','text':'演示完成'}]},))
            return
        call = ToolCall(f'demo-{self.step}', f'call-{self.step}', name, json.dumps(args))
        yield ModelFrame('response_finished', status='completed', calls=(call,))

async def main():
    with tempfile.TemporaryDirectory(prefix='cagent-demo-') as temp:
        root = Path(temp) / 'repo'
        shutil.copytree(Path(__file__).parent / 'bug_demo', root)
        config = AgentConfig(model='offline-demo', state_dir=Path(temp) / 'state')
        store = JsonlSessionStore(config.state_dir, root)
        policy = Policy()
        policy.approve_for_session([sys.executable, '-m', 'pytest', '-q', '-o', 'addopts='], root)
        async def emit(kind, payload):
            pass
        snapshots = SnapshotStore(root, store.directory)
        context = ToolContext(root, '', snapshots, policy, emit, Redactor(), store.directory)
        engine = AgentEngine(DemoProvider(root), toolset(), store, context, config)
        renderer = TerminalRenderer()
        try:
            async for event in engine.run_turn('修复 average([]) 失败的问题并验证', asyncio.Event()):
                renderer.handle(event)
            print(snapshots.diff())
        finally:
            store.close()

if __name__ == '__main__':
    asyncio.run(main())
