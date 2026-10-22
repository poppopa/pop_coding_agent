"""并发排空两个管道；超时和用户中断共享进程组回收流程。"""
from __future__ import annotations
from pathlib import Path
import asyncio
import codecs
import os
import signal
import uuid
from typing import Literal
from pydantic import Field, field_validator
from coding_agent.domain import ToolResult
from coding_agent.tools.base import Tool, ToolInput, ToolContext

class CommandInput(ToolInput):
    argv: list[str] = Field(min_length=1, max_length=200)
    cwd: str = '.'
    timeout_seconds: int = Field(default=120, ge=1, le=600)
    purpose: Literal['inspect', 'verify', 'other'] = 'other'

    @field_validator('argv')
    @classmethod
    def valid_argv(cls, value):
        if any('\x00' in s for s in value) or not value[0].strip():
            raise ValueError('命令参数不能包含 NUL 或空程序名')
        return value

class RunCommandTool(Tool):
    name = 'run_command'
    description = '使用 argv 数组执行明确的本地命令。没有隐式 shell；命令需用户授权。'
    input_type = CommandInput
    mutating = True

    async def execute(self, arguments: dict, context: ToolContext, cancel_event: asyncio.Event) -> ToolResult:
        args = CommandInput.model_validate(arguments)
        cwd = (context.root / args.cwd).resolve()
        if not cwd.is_relative_to(context.root.resolve()):
            return ToolResult.fail(context.call_id, 'PATH_OUTSIDE_ROOT', '命令工作目录超出仓库')
        if not cwd.is_dir():
            return ToolResult.fail(context.call_id, 'FILE_NOT_FOUND', '工作目录不存在')
        if context.policy.evaluate(args.argv, cwd) != 'allow':
            return ToolResult.fail(context.call_id, 'APPROVAL_REQUIRED', '命令尚未获得授权')
        if cancel_event.is_set():
            return ToolResult.fail(context.call_id, 'COMMAND_CANCELLED', '命令启动前已取消')
        env = {k: v for k, v in os.environ.items() if k not in context.filtered_env_names}
        directory = context.state_dir / 'outputs'
        output_path = directory / (uuid.uuid4().hex + '.log')
        output = None
        try:
            # 日志初始化完整成功后才启动命令，失败时只需关闭已打开的日志。
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            output = output_path.open('w', encoding='utf-8')
            output_path.chmod(0o600)
            proc = await asyncio.create_subprocess_exec(*args.argv, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        except OSError as exc:
            if output is not None:
                output.close()
            return ToolResult.fail(context.call_id, 'COMMAND_FAILED', context.redactor.text(str(exc)))
        except BaseException:
            if output is not None:
                output.close()
            raise
        tails = {'stdout': '', 'stderr': ''}
        counts = {'stdout': 0, 'stderr': 0}
        tail_chunks: list[tuple[str, str]] = []
        logged = 0
        max_log = 1024 * 1024
        published = 0
        limit = 32768

        async def pump(reader, stream):
            nonlocal logged
            decoder = codecs.getincrementaldecoder('utf-8')('replace')
            pending = ''
            # 保留最长凭证长度的尾部，防止 Key 被不同管道 chunk 拆开泄漏。
            overlap = max((len(s) for s in context.redactor.secrets), default=0) + 128
            async def publish(text):
                nonlocal logged, published
                safe = context.redactor.text(text)
                encoded = safe.encode('utf-8')
                counts[stream] += len(encoded)
                tail_chunks.append((stream, safe))
                total_tail = sum(len(part.encode('utf-8')) for _, part in tail_chunks)
                while total_tail > limit and tail_chunks:
                    first_stream, first_text = tail_chunks.pop(0)
                    original = first_text.encode('utf-8')
                    overflow = total_tail - limit
                    if len(original) > overflow:
                        remainder = original[overflow:].decode('utf-8', errors='ignore')
                        tail_chunks.insert(0, (first_stream, remainder))
                        total_tail -= len(original) - len(remainder.encode('utf-8'))
                    else:
                        total_tail -= len(original)
                tails['stdout'] = ''.join(part for name, part in tail_chunks if name == 'stdout')
                tails['stderr'] = ''.join(part for name, part in tail_chunks if name == 'stderr')
                if logged < max_log:
                    record = (f'[{stream}] ' + safe).encode('utf-8')
                    part = record[:max_log - logged].decode('utf-8', errors='ignore')
                    output.write(part)
                    output.flush()
                    logged += len(part.encode('utf-8'))
                if published < max_log:
                    visible = encoded[:max_log-published].decode('utf-8', errors='ignore')
                    published += len(visible.encode('utf-8'))
                    await context.emit('tool_output', {'call_id': context.call_id, 'stream': stream, 'text': visible})
                    if published >= max_log:
                        await context.emit('tool_output', {'call_id': context.call_id, 'stream': stream, 'text': '\n[输出展示达到上限，继续排空并保留尾部]\n'})
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    pending += decoder.decode(b'', final=True)
                    if pending:
                        await publish(pending)
                    return
                pending += decoder.decode(chunk)
                if len(pending) > overlap:
                    cut = len(pending) - overlap
                    # 收集全部匹配（含重叠），逆序移动分界，避免切开已扫描的匹配。
                    matches = []
                    for secret in context.redactor.secrets:
                        start = pending.find(secret)
                        while start >= 0:
                            matches.append((start, start + len(secret)))
                            start = pending.find(secret, start + 1)
                    for start, end in sorted(matches, reverse=True):
                        if start < cut < end:
                            cut = start
                    if cut:
                        await publish(pending[:cut])
                        pending = pending[cut:]

        async def terminate_group():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            await asyncio.sleep(.1)
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except asyncio.TimeoutError:
                pass
            # 父进程退出不能代表其子进程已结束，仍检查整个进程组。
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

        pumps = [asyncio.create_task(pump(proc.stdout, 'stdout')), asyncio.create_task(pump(proc.stderr, 'stderr'))]
        waiter = asyncio.create_task(proc.wait())
        cancelled = asyncio.create_task(cancel_event.wait())
        timed_out = False
        interrupted = False
        try:
            done, _ = await asyncio.wait([waiter, cancelled], timeout=args.timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done or not done:
                interrupted = cancelled in done
                timed_out = not done
                await terminate_group()
            await waiter
            # 正常退出也清理命令遗留的进程组，含已重定向管道的子进程。
            await terminate_group()
            # 命令退出后若遗留持有管道的子进程，限制排空等待并清理。
            try:
                await asyncio.wait_for(asyncio.gather(*pumps), 2)
            except asyncio.TimeoutError:
                await terminate_group()
        except asyncio.CancelledError:
            interrupted = True
            await terminate_group()
            await asyncio.gather(*pumps, return_exceptions=True)
        except Exception:
            await terminate_group()
            raise
        finally:
            cancelled.cancel()
            for task in pumps + [waiter]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pumps, waiter, cancelled, return_exceptions=True)
            output.close()
        data = {'exit_code': proc.returncode, 'stdout': tails['stdout'], 'stderr': tails['stderr'],
            'timed_out': timed_out, 'cancelled': interrupted, 'output_path': str(output_path),
            'purpose': args.purpose}
        truncated = sum(counts.values()) > limit or logged >= max_log
        if interrupted or timed_out:
            from coding_agent.domain import ToolError
            code = 'COMMAND_CANCELLED' if interrupted else 'COMMAND_TIMEOUT'
            return ToolResult(context.call_id, False, data, ToolError(code, '命令已中断' if interrupted else '命令超时'), truncated)
        if proc.returncode:
            from coding_agent.domain import ToolError
            return ToolResult(context.call_id, False, data, ToolError('COMMAND_FAILED', f'命令退出码 {proc.returncode}'), truncated)
        return ToolResult(context.call_id, True, data, truncated=truncated)
