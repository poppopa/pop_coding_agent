"""显示 Agent 快照差异，或显式包含用户未提交修改的 Git 差异。"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from coding_agent.domain import ToolResult
from coding_agent.storage.snapshots import make_snapshot, unified_diff
from coding_agent.tools.base import Tool, ToolContext, ToolInput
from coding_agent.tools.files import _stop_process, checked_path, file_failure
from coding_agent.tools.paths import FileToolError, read_bytes

MAX_DIFF_BYTES = 1024 * 1024


class ShowDiffInput(ToolInput):
    scope: Literal['agent', 'git'] = 'agent'
    path: str | None = Field(default=None, min_length=1)


async def _git(root: Path, executable: str, arguments: list[str],
               cancel_event: asyncio.Event) -> tuple[int, bytes, bool]:
    """Git 用固定 argv，不执行 shell、外部 diff 或 textconv 驱动。

    输出超过上限仍继续排空，避免管道死锁；所有子进程都有超时和
    取消清理。Git 差异只读，不会调用 reset、checkout 或 clean。
    """
    if cancel_event.is_set():
        raise FileToolError('TOOL_CANCELLED', '差异查询已取消')
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            executable, '--no-pager', *arguments, cwd=root,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        output = bytearray()
        truncated = False
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '差异查询已取消')
            if asyncio.get_running_loop().time() >= deadline:
                raise FileToolError('GIT_DIFF_FAILED', 'Git 差异查询超过 30 秒')
            try:
                chunk = await asyncio.wait_for(process.stdout.read(8192), 0.1)
            except TimeoutError:
                continue
            if not chunk:
                break
            available = MAX_DIFF_BYTES - len(output)
            output.extend(chunk[:available])
            truncated = truncated or len(chunk) > available
        remaining = max(0.01, deadline - asyncio.get_running_loop().time())
        try:
            code = await asyncio.wait_for(process.wait(), remaining)
        except TimeoutError:
            raise FileToolError('GIT_DIFF_FAILED', 'Git 差异查询超过 30 秒') from None
        return code, bytes(output), truncated
    except OSError:
        raise FileToolError('GIT_UNAVAILABLE', '无法启动 Git') from None
    finally:
        if process is not None:
            await _stop_process(process)


async def _git_diff(context: ToolContext, path: str | None,
                    cancel_event: asyncio.Event) -> tuple[str, bool, int]:
    executable = shutil.which('git')
    if executable is None:
        raise FileToolError('GIT_UNAVAILABLE', '系统中未安装 Git')
    root = Path(context.root).resolve()
    code, inside, _ = await _git(root, executable, ['rev-parse', '--is-inside-work-tree'], cancel_event)
    if code != 0 or inside.strip() != b'true':
        raise FileToolError('NOT_GIT_REPOSITORY', '仓库根目录不在 Git 工作区内')
    selected = None if path is None else checked_path(context, path).relative_to(root).as_posix()
    # 不把模型提供的 glob/冒号当作 Git pathspec 表达式。列表始终限定
    # 到当前 root，之后再检查真实路径和默认敏感文件忽略规则。
    listing_scope = ['--', f':(literal){selected}'] if selected is not None else ['--', '.']
    code, listing, truncated = await _git(
        root, executable, ['ls-files', '--cached', '--others', '--exclude-standard', '-z',
                           *listing_scope], cancel_event,
    )
    if code != 0:
        raise FileToolError('GIT_DIFF_FAILED', '无法枚举 Git 文件')
    code, _, _ = await _git(root, executable, ['rev-parse', '--verify', 'HEAD'], cancel_event)
    has_head = code == 0
    names = listing.split(b'\x00')
    if names[-1]:
        # 被字节上限截断的最后一个名字不能作为真实文件路径使用。
        names.pop()
        truncated = True
    if has_head:
        # 已暂存删除的路径已不在 index，ls-files --cached 不会列出。
        # 加入 HEAD 差异的路径，确保删除与重命名的旧路径仍可展示。
        code, changed, cut = await _git(root, executable, [
            'diff', '--no-ext-diff', '--no-textconv', '--name-only', '-z',
            'HEAD', *listing_scope,
        ], cancel_event)
        if code != 0:
            raise FileToolError('GIT_DIFF_FAILED', '无法枚举 Git 修改路径')
        names.extend(changed.split(b'\x00')[:-1])
        truncated = truncated or cut
    paths = []
    skipped = 0
    for name in sorted(set(names)):
        if not name:
            continue
        relative = os.fsdecode(name)
        try:
            checked_path(context, relative)
        except FileToolError:
            skipped += 1
            continue
        paths.append(relative)
    output = []
    size = 0
    if has_head and paths:
        # 分批 argv 防止大型仓库触及系统参数长度上限。HEAD 对当前
        # 工作区的差异同时包含已暂存和未暂存的净变化。
        for start in range(0, len(paths), 64):
            code, patch, cut = await _git(root, executable, [
                'diff', '--no-ext-diff', '--no-textconv', '--no-color', 'HEAD', '--',
                *(f':(literal){name}' for name in paths[start:start + 64]),
            ], cancel_event)
            if code != 0:
                raise FileToolError('GIT_DIFF_FAILED', 'Git 差异生成失败')
            remaining = MAX_DIFF_BYTES - size
            output.append(patch[:remaining].decode('utf-8', errors='replace'))
            size += min(len(patch), remaining)
            truncated = truncated or cut or len(patch) > remaining
            if size >= MAX_DIFF_BYTES:
                truncated = True
                break
    # git diff 不展示未跟踪文件，单独补充其创建差异。无 HEAD 的新
    # 仓库以空树为基线生成所有当前文件，避免 rev-parse HEAD 错误。
    if size < MAX_DIFF_BYTES:
        code, untracked, cut = await _git(root, executable, [
            'ls-files', '--others', '--exclude-standard', '-z', *listing_scope,
        ], cancel_event)
        if code != 0:
            raise FileToolError('GIT_DIFF_FAILED', '无法枚举 Git 未跟踪文件')
        truncated = truncated or cut
        new_names = set(untracked.split(b'\x00')[:-1]) if has_head else None
        for relative in paths:
            if new_names is not None and os.fsencode(relative) not in new_names:
                continue
            try:
                resolved = checked_path(context, relative)
                content, mode = read_bytes(resolved)
                patch = unified_diff(relative, make_snapshot(None), make_snapshot(content, mode))
            except FileToolError as exc:
                if exc.code == 'FILE_NOT_FOUND':
                    continue
                skipped += 1
                continue
            encoded = patch.encode('utf-8', errors='replace')
            remaining = MAX_DIFF_BYTES - size
            output.append(encoded[:remaining].decode('utf-8', errors='replace'))
            size += min(len(encoded), remaining)
            if len(encoded) > remaining:
                truncated = True
                break
            await asyncio.sleep(0)
    return ''.join(output), truncated, skipped


class ShowDiffTool(Tool):
    name = 'show_diff'
    description = '显示 Agent 快照修改；git 范围包括用户已有的已暂存、未暂存和未跟踪修改'
    input_type = ShowDiffInput

    async def execute(self, arguments: dict, context: ToolContext,
                      cancel_event: asyncio.Event) -> ToolResult:
        try:
            args = self.input_type.model_validate(arguments)
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '差异查询已取消')
            if args.scope == 'agent':
                if args.path is not None:
                    checked_path(context, args.path, for_write=True)
                diff = context.snapshots.diff(args.path)
                encoded = diff.encode('utf-8', errors='replace')
                truncated = len(encoded) > MAX_DIFF_BYTES
                diff = encoded[:MAX_DIFF_BYTES].decode('utf-8', errors='replace')
                note = '仅展示文件工具记录的 Agent 修改；用户在动作之间的编辑不纳入差异'
                skipped = 0
            else:
                diff, truncated, skipped = await _git_diff(context, args.path, cancel_event)
                note = 'Git 范围包括用户已有的未提交修改（已暂存、未暂存及未跟踪文件），遵循默认敏感文件忽略规则'
            return ToolResult(context.call_id, True, context.redactor.value({
                'scope': args.scope, 'path': args.path, 'diff': diff, 'note': note,
                'includes_user_changes': args.scope == 'git',
                'skipped_files': skipped, 'truncated': truncated,
            }), truncated=truncated)
        except ValidationError:
            return ToolResult.fail(context.call_id, 'INVALID_ARGUMENTS', 'show_diff 参数不符合约束')
        except FileToolError as exc:
            return file_failure(context, exc)
