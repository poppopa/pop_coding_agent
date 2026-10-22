"""输入与取消是用户控制入口；终端不参与模型的工具决策。"""
from __future__ import annotations
import asyncio
import signal
import shlex
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.syntax import Syntax
from coding_agent.ui.renderer import TerminalRenderer

HELP = '''Enter 提交，Alt+Enter 换行；Ctrl+C 中断任务或清空输入，Ctrl+D 在空输入时退出。
/help 帮助，/mode analysis|execute 切换模式，/diff 查看 Agent 修改，
/context 查看预算，/compact 整理历史，/exit 退出。'''

class SafeHistory(FileHistory):
    def __init__(self, filename, redactor):
        self.redactor = redactor
        super().__init__(filename)
    def store_string(self, string):
        super().store_string(self.redactor.text(string))

async def interactive(engine, config, console=None):
    console = console or Console()
    renderer = TerminalRenderer(console)
    bindings = KeyBindings()
    @bindings.add('enter')
    def submit(event):
        event.current_buffer.validate_and_handle()
    @bindings.add('escape', 'enter')
    def newline(event):
        event.current_buffer.insert_text('\n')
    @bindings.add('c-c')
    def clear(event):
        event.current_buffer.reset()
    @bindings.add('c-d')
    def leave(event):
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError())
    session = PromptSession(multiline=True, key_bindings=bindings,
        history=SafeHistory(str(engine.store.directory / 'input_history'), engine.context.redactor))
    console.print('个人编程助手 · Ark · /help 查看帮助', style='cyan')
    console.print(f'会话：{engine.session_id}', markup=False)

    async def approval(argv, cwd):
        console.print('命令：' + shlex.join(argv), markup=False)
        console.print('目录：' + str(cwd), markup=False)
        try:
            with patch_stdout():
                answer = await PromptSession().prompt_async('允许一次 [y] / 本会话相同命令 [s] / 拒绝 [n]：', handle_sigint=False)
        except (EOFError, KeyboardInterrupt):
            return 'deny'
        return {'y': 'once', 's': 'session'}.get(answer.strip().lower(), 'deny')
    engine.approval = approval
    while True:
        try:
            with patch_stdout():
                prompt = await session.prompt_async(f'{engine.context.policy.mode} > ')
        except EOFError:
            return 0
        except KeyboardInterrupt:
            continue
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.startswith('/'):
            if prompt in ('/exit', '/quit'):
                return 0
            if prompt == '/help':
                console.print(HELP, markup=False)
            elif prompt.startswith('/mode '):
                mode = prompt.split(maxsplit=1)[1]
                if mode in ('analysis', 'execute'):
                    engine.context.policy.mode = mode
                    console.print('模式：' + mode)
                else:
                    console.print('模式必须是 analysis 或 execute')
            elif prompt == '/diff':
                console.print(Syntax(engine.context.snapshots.diff() or '没有 Agent 文件修改', 'diff'))
            elif prompt == '/context':
                schemas = engine.registry.schemas(engine.context.policy.mode)
                console.print(f'请求保守估算：{engine.manager.request_size(engine.history, schemas)} / '
                              f'可用输入预算 {engine.manager.input_budget}')
            elif prompt == '/compact':
                cancel = asyncio.Event()
                loop = asyncio.get_running_loop()
                previous = signal.getsignal(signal.SIGINT)
                loop.add_signal_handler(signal.SIGINT, cancel.set)
                try:
                    changed = await engine.compact(cancel)
                    console.print('历史已整理，完整日志保留。' if changed else '当前没有可整理的旧轮次，无需整理。')
                except asyncio.CancelledError:
                    console.print('历史整理已取消，保留原历史。')
                except Exception as exc:
                    console.print(engine.context.redactor.text(str(exc)), markup=False)
                finally:
                    loop.remove_signal_handler(signal.SIGINT)
                    signal.signal(signal.SIGINT, previous)
            else:
                console.print('未知本地命令，输入 /help 查看。')
            continue
        cancel = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGINT)
        loop.add_signal_handler(signal.SIGINT, cancel.set)
        try:
            with patch_stdout():
                async for event in engine.run_turn(prompt, cancel):
                    renderer.handle(event)
        finally:
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, previous)
