"""CLI 装配依赖；真正执行逻辑在 AgentEngine，脚本与终端共用它。"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from pydantic import ValidationError
from coding_agent.config import AgentConfig, load_config
from coding_agent.domain import ToolResult
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.storage.session import JsonlSessionStore, SessionError
from coding_agent.tools.base import ToolContext
from coding_agent.ui.renderer import JsonRenderer, TerminalRenderer

EXIT_CODES = {'completed': 0, 'failed': 1, 'approval_required': 3, 'interrupted': 130}

def parser():
    p = argparse.ArgumentParser(prog='cagent', description='个人编码与学习展示用的 Ark 编程助手')
    p.add_argument('--model', help='Ark 模型标识或接入点 ID')
    p.add_argument('--mode', choices=['analysis', 'execute'])
    p.add_argument('--resume', help='恢复本地会话')
    p.add_argument('--root', type=Path, default=Path.cwd(), help='工作目录')
    p.add_argument('--config', type=Path, help='用户 TOML 配置路径')
    p.add_argument('--state-dir', type=Path, help='本地会话保存目录')
    sub = p.add_subparsers(dest='command')
    execute = sub.add_parser('exec', help='单次任务，支持 JSONL 输出')
    execute.add_argument('prompt')
    execute.add_argument('--json', action='store_true', dest='json_output')
    execute.add_argument('--allow-command', action='append', default=[], metavar='JSON_ARGV', help='预先允许当前仓库内精确命令，例如 ["python","-V"]')
    sub.add_parser('sessions', help='列出本地会话')
    sub.add_parser('doctor', help='离线检查配置与依赖，不请求模型')
    return p

def toolset():
    from coding_agent.tools.files import ReadFileTool, FindFilesTool, SearchTextTool
    from coding_agent.tools.patch import ApplyPatchTool
    from coding_agent.tools.diff import ShowDiffTool
    from coding_agent.tools.command import RunCommandTool
    return [ReadFileTool(), FindFilesTool(), SearchTextTool(), ApplyPatchTool(), RunCommandTool(), ShowDiffTool()]

async def run_async(args, config, dependencies=None):
    dependencies = dependencies or {}
    if args.command == 'doctor':
        from importlib.metadata import version
        checks = {'python': sys.version.split()[0], 'model_configured': bool(config.model and config.model != '__unconfigured__'),
            'key_present': bool(os.environ.get('ARK_API_KEY')), 'api': config.api,
            'dependencies': {name: version(name) for name in ('volcengine-python-sdk', 'prompt-toolkit', 'rich', 'pydantic')}}
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if checks['key_present'] and checks['model_configured'] else 2
    if args.command == 'sessions':
        print(json.dumps(JsonlSessionStore.list_sessions(config.state_dir), ensure_ascii=False, indent=2))
        return 0
    if not config.model and 'provider' not in dependencies:
        raise ValueError('请配置 model，或通过 --model / ARK_MODEL 指定模型标识')
    secrets = [os.environ[name] for name in config.filtered_env_names if os.environ.get(name)]
    redactor = Redactor(secrets)
    if 'provider' in dependencies:
        provider = dependencies['provider']
    else:
        from coding_agent.providers.ark import ArkProvider
        provider = ArkProvider(config=config)
    store = None
    try:
        store = JsonlSessionStore(config.state_dir, args.root, args.resume, redactor)
        from coding_agent.storage.snapshots import SnapshotStore
        snapshots = SnapshotStore(args.root.resolve(), store.directory)
        policy = Policy(config.mode)
        for specification in getattr(args, 'allow_command', []):
            argv = json.loads(specification)
            if not isinstance(argv, list) or not argv or not all(isinstance(s, str) for s in argv):
                raise ValueError('--allow-command 必须是非空 JSON 字符串数组')
            policy.approve_for_session(argv, args.root)
        async def emit(kind, payload):
            pass  # 内核为每次调用注入事件队列，此默认回调仅用于装配。
        context = ToolContext(args.root.resolve(), '', snapshots, policy, emit, redactor, store.directory, tuple(config.filtered_env_names))
        engine = AgentEngine(provider, dependencies.get('tools') or toolset(), store, context, config)
        if args.command != 'exec':
            from coding_agent.ui.terminal import interactive
            return await interactive(engine, config)
        renderer = JsonRenderer() if args.json_output else TerminalRenderer()
        cancel = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGINT)
        loop.add_signal_handler(signal.SIGINT, cancel.set)
        code = 1
        try:
            async for event in engine.run_turn(args.prompt, cancel):
                renderer.handle(event)
                if event.type == 'turn_finished':
                    code = EXIT_CODES[event.payload['status']]
        finally:
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, previous)
        return code
    finally:
        if store:
            store.close()
        if hasattr(provider, 'close'):
            await provider.close()

def run_cli(argv=None, dependencies=None):
    args = parser().parse_args(argv)
    redactor = Redactor([os.environ.get('ARK_API_KEY', '')])
    try:
        if not args.root.is_dir():
            raise ValueError('工作目录不存在')
        args.root = args.root.resolve()
        config = load_config(args.config, {'model': args.model, 'mode': args.mode, 'state_dir': args.state_dir}, allow_unconfigured=args.command in ('doctor', 'sessions') or bool(dependencies))
        return asyncio.run(run_async(args, config, dependencies))
    except (ValueError, ValidationError, SessionError, OSError) as exc:
        print(redactor.text(str(exc)), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print('启动失败：' + redactor.text(str(exc)), file=sys.stderr)
        return 1

def main():
    raise SystemExit(run_cli())

if __name__ == '__main__':
    main()
