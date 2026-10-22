"""有限文件读取、路径查找与字面文本搜索，统一沿用公共工具契约。"""
from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import Field, ValidationError

from coding_agent.domain import ToolResult
from coding_agent.tools.base import Tool, ToolContext, ToolInput
from coding_agent.tools.paths import (
    IGNORED_DIRECTORIES, SENSITIVE_PATTERNS, FileToolError, decode_text,
    iter_files, matches_glob, read_bytes, safe_path, sha256,
)


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=400)


class FindFilesInput(ToolInput):
    pattern: str = Field(min_length=1)
    max_results: int = Field(default=100, ge=1, le=500)


class SearchTextInput(ToolInput):
    query: str = Field(min_length=1)
    glob: str = Field(default='**/*', min_length=1)
    max_results: int = Field(default=100, ge=1, le=500)


def checked_path(context: ToolContext, path: str, *, for_write: bool = False) -> Path:
    resolved = safe_path(context.root, path, for_write=for_write)
    if resolved.is_relative_to(Path(context.state_dir).resolve()):
        raise FileToolError('PERMISSION_DENIED', '文件工具不能访问本会话的存储目录')
    return resolved


def file_failure(context: ToolContext, error: FileToolError) -> ToolResult:
    return ToolResult.fail(context.call_id, error.code, error.message, error.retryable)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def _candidates(context: ToolContext, pattern: str,
                      cancel_event: asyncio.Event) -> AsyncIterator[Path]:
    """rg 只提供候选路径，安全边界仍由 Python 校验；缺少 rg 时回退。

    以 argv 调用、--no-config 关闭个人配置、--threads 1 固定单工作线程。
    使用 NUL 分隔处理含换行的文件名，流式消费避免大型仓库积累全部
    输出。提前满足结果上限时关闭生成器，同时回收 rg 子进程。
    """
    executable = shutil.which('rg')
    if executable is None:
        for path in iter_files(context.root, pattern, excluded=(context.state_dir,)):
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件查找已取消')
            yield path
            await asyncio.sleep(0)
        return
    argv = [executable, '--no-config', '--files', '--hidden', '--no-ignore',
            '--null', '--threads', '1']
    for name in sorted(IGNORED_DIRECTORIES):
        argv.extend(['--glob', f'!{name}/**', '--glob', f'!**/{name}/**'])
    for sensitive in SENSITIVE_PATTERNS:
        argv.extend(['--glob', f'!{sensitive}', '--glob', f'!**/{sensitive}/**'])
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=context.root, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        buffer = b''
        while True:
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件查找已取消')
            # 有超时的分块读取让取消能在没有任何候选文件时被观察到。
            try:
                chunk = await asyncio.wait_for(process.stdout.read(8192), 0.1)
            except TimeoutError:
                continue
            if not chunk:
                break
            buffer += chunk
            while b'\x00' in buffer:
                raw, buffer = buffer.split(b'\x00', 1)
                relative = os.fsdecode(raw)
                if not matches_glob(relative, pattern):
                    continue
                try:
                    path = checked_path(context, relative)
                    if not path.is_file():
                        continue
                except (FileToolError, OSError):
                    continue
                yield Path(context.root) / relative
        returncode = await process.wait()
        if returncode not in (0, 1):
            raise FileToolError('FILE_SEARCH_FAILED', 'rg 文件查找失败')
    except OSError:
        raise FileToolError('FILE_SEARCH_FAILED', '无法启动文件查找进程') from None
    finally:
        if process is not None:
            await _stop_process(process)


class ReadFileTool(Tool):
    name = 'read_file'
    description = '读取仓库内 UTF-8 文件的指定行，返回完整文件 SHA-256；文件最大 1 MiB'
    input_type = ReadFileInput

    async def execute(self, arguments: dict, context: ToolContext,
                      cancel_event: asyncio.Event) -> ToolResult:
        try:
            args = self.input_type.model_validate(arguments)
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件读取已取消')
            path = checked_path(context, args.path)
            content, _ = read_bytes(path)
            text = decode_text(content)
            # 指纹来自完整原始字节，不能用行范围或脱敏文本计算，否则
            # expected_sha256 无法保护文件的未显示部分。splitlines 保留
            # 原换行符，CRLF 文件也不经过隐式换行转换。
            lines = text.splitlines(keepends=True)
            start = args.start_line - 1
            selected = lines[start:start + args.max_lines]
            truncated = start > 0 or start + len(selected) < len(lines)
            data = {
                'path': path.relative_to(Path(context.root).resolve()).as_posix(),
                'start_line': args.start_line,
                'end_line': args.start_line + len(selected) - 1 if selected else None,
                'total_lines': len(lines), 'text': ''.join(selected),
                'lines': [{'line_number': args.start_line + i, 'text': line}
                          for i, line in enumerate(selected)],
                'sha256': sha256(content), 'size_bytes': len(content), 'truncated': truncated,
            }
            return ToolResult(context.call_id, True, context.redactor.value(data), truncated=truncated)
        except ValidationError:
            return ToolResult.fail(context.call_id, 'INVALID_ARGUMENTS', 'read_file 参数不符合约束')
        except FileToolError as exc:
            return file_failure(context, exc)


class FindFilesTool(Tool):
    name = 'find_files'
    description = '按 glob 查找仓库文件，跳过敏感文件、依赖与构建目录'
    input_type = FindFilesInput

    async def execute(self, arguments: dict, context: ToolContext,
                      cancel_event: asyncio.Event) -> ToolResult:
        candidates = None
        try:
            args = self.input_type.model_validate(arguments)
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件查找已取消')
            paths = []
            truncated = False
            candidates = _candidates(context, args.pattern, cancel_event)
            async for path in candidates:
                if len(paths) == args.max_results:
                    truncated = True
                    break
                paths.append(path.relative_to(context.root).as_posix())
            return ToolResult(context.call_id, True, context.redactor.value({
                'paths': paths, 'count': len(paths), 'truncated': truncated,
            }), truncated=truncated)
        except ValidationError:
            return ToolResult.fail(context.call_id, 'INVALID_ARGUMENTS', 'find_files 参数不符合约束')
        except FileToolError as exc:
            return file_failure(context, exc)
        finally:
            if candidates is not None:
                await candidates.aclose()


class SearchTextTool(Tool):
    name = 'search_text'
    description = '在仓库 UTF-8 文件中逐行按字面文本搜索，结果默认 100、最多 500'
    input_type = SearchTextInput

    async def execute(self, arguments: dict, context: ToolContext,
                      cancel_event: asyncio.Event) -> ToolResult:
        candidates = None
        try:
            args = self.input_type.model_validate(arguments)
            if '\n' in args.query or '\r' in args.query:
                raise FileToolError('INVALID_ARGUMENTS', 'search_text 使用单行字面文本查询')
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文本搜索已取消')
            matches = []
            truncated = False
            skipped = 0
            candidates = _candidates(context, args.glob, cancel_event)
            async for candidate in candidates:
                try:
                    path = checked_path(context, candidate.as_posix())
                    content, _ = read_bytes(path)
                    text = decode_text(content)
                except FileToolError:
                    # 目录搜索遇到超大、二进制、不可访问或恰好消失的
                    # 文件时继续；显式 read_file 会返回该文件的错误。
                    skipped += 1
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if args.query not in line:
                        continue
                    if len(matches) == args.max_results:
                        truncated = True
                        break
                    offset = line.index(args.query)
                    # 长单行文件也不能放大模型输出；截取匹配点附近片段。
                    begin = max(0, offset - 160)
                    end = min(len(line), offset + max(len(args.query), 160))
                    snippet = line[begin:end]
                    if len(snippet) > 2000:
                        snippet = snippet[:2000]
                    matches.append({
                        'path': candidate.relative_to(context.root).as_posix(),
                        'line_number': number, 'text': snippet,
                        'truncated': begin > 0 or begin + len(snippet) < len(line),
                    })
                if truncated:
                    break
                await asyncio.sleep(0)
            return ToolResult(context.call_id, True, context.redactor.value({
                'matches': matches, 'count': len(matches), 'skipped_files': skipped,
                'truncated': truncated,
            }), truncated=truncated)
        except ValidationError:
            return ToolResult.fail(context.call_id, 'INVALID_ARGUMENTS', 'search_text 参数不符合约束')
        except FileToolError as exc:
            return file_failure(context, exc)
        finally:
            if candidates is not None:
                await candidates.aclose()
