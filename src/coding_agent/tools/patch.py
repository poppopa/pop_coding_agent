"""单文件精确补丁：先验完整指纹和全部编辑，再一次发布最终内容。"""
from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from coding_agent.domain import ToolError, ToolResult
from coding_agent.storage.snapshots import make_snapshot
from coding_agent.tools.base import Tool, ToolContext, ToolInput
from coding_agent.tools.files import checked_path, file_failure
from coding_agent.tools.paths import (
    MAX_FILE_BYTES, FileToolError, atomic_write, decode_text, delete_file,
    read_bytes, sha256,
)


class PatchEdit(ToolInput):
    old_text: str = Field(min_length=1)
    new_text: str


class ApplyPatchInput(ToolInput):
    path: str = Field(min_length=1)
    operation: Literal['create', 'update', 'delete']
    expected_sha256: str | None = Field(default=None, pattern=r'^[0-9a-fA-F]{64}$')
    edits: list[PatchEdit] | None = None
    content: str | None = None

    @model_validator(mode='after')
    def validate_operation(self) -> ApplyPatchInput:
        # 不允许操作无关字段被静默忽略，避免模型以为提交的 edits 或
        # content 已经应用。create 的空字符串是合法的空文件内容。
        if self.operation == 'create':
            if self.content is None or self.expected_sha256 is not None or self.edits:
                raise ValueError('create 需要 content，其他修改字段必须为空')
        elif self.operation == 'update':
            if self.expected_sha256 is None or not self.edits or self.content is not None:
                raise ValueError('update 需要 expected_sha256 和非空 edits')
        elif self.expected_sha256 is None or self.edits or self.content is not None:
            raise ValueError('delete 需要 expected_sha256，其他修改字段必须为空')
        return self


def apply_edits(text: str, edits: list[PatchEdit]) -> str:
    """全部片段基于同一份原始内容定位，禁止顺序 replace 造成连锁匹配。

    唯一性包含重叠出现：在 aaa 中搜索 aa 有两个位置，str.count 却
    只统计一个。所有区间确定且互不相交后，按位置一次拼接，任何
    无匹配、重复匹配或重叠都会在写入前结束。
    """
    located = []
    for edit in edits:
        begin = text.find(edit.old_text)
        if begin < 0 or text.find(edit.old_text, begin + 1) >= 0:
            raise FileToolError('PATCH_NOT_UNIQUE', '每个 old_text 必须在当前原始内容中恰好匹配一次')
        located.append((begin, begin + len(edit.old_text), edit.new_text))
    located.sort(key=lambda item: item[0])
    cursor = 0
    output = []
    for begin, end, replacement in located:
        if begin < cursor:
            raise FileToolError('PATCH_NOT_UNIQUE', '多个 edits 的原始片段发生重叠')
        output.extend([text[cursor:begin], replacement])
        cursor = end
    output.append(text[cursor:])
    return ''.join(output)


class ApplyPatchTool(Tool):
    name = 'apply_patch'
    description = '按 SHA-256 校验并创建、精确更新或删除单个文件；保存 Agent 修改前后快照'
    input_type = ApplyPatchInput
    mutating = True

    async def execute(self, arguments: dict, context: ToolContext,
                      cancel_event: asyncio.Event) -> ToolResult:
        try:
            args = self.input_type.model_validate(arguments)
            if getattr(context.policy, 'mode', 'execute') == 'analysis':
                raise FileToolError('PERMISSION_DENIED', '分析模式不允许修改文件')
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件补丁已取消')
            path = checked_path(context, args.path, for_write=True)
            # 已知模型凭证不能进入任何持久快照；先读取原始字节、
            # 计算目标内容并检查凭证，再构造 before/after。这里不
            # 脱敏源代码，否则恢复会破坏原文件。用户私有代码允许
            # 无损保存，凭证命中则拒绝整次动作且不执行文件写入。
            try:
                original, original_mode = read_bytes(path)
            except FileToolError as exc:
                if exc.code != 'FILE_NOT_FOUND':
                    raise
                original, original_mode = None, None
            if args.operation == 'create':
                if original is not None:
                    raise FileToolError('FILE_CHANGED', 'create 目标已经存在，未覆盖文件', True)
                content = args.content
                mode = 0o644
            else:
                if original is None:
                    raise FileToolError('FILE_NOT_FOUND', '修改或删除的目标文件不存在')
                if sha256(original) != args.expected_sha256.lower():
                    raise FileToolError('FILE_CHANGED', '文件指纹已改变，请重新读取后重试', True)
                text = decode_text(original)
                content = apply_edits(text, args.edits) if args.operation == 'update' else None
                mode = original_mode
            try:
                encoded = None if content is None else content.encode('utf-8')
            except UnicodeEncodeError:
                raise FileToolError('UNSUPPORTED_FILE', '补丁内容不能编码为 UTF-8') from None
            if encoded is not None:
                if len(encoded) > MAX_FILE_BYTES:
                    raise FileToolError('UNSUPPORTED_FILE', '修改后的文件超过 1 MiB 上限')
                decode_text(encoded)
            for secret in context.redactor.secrets:
                if not secret:
                    continue
                secret_bytes = secret.encode('utf-8')
                if any(secret_bytes in value for value in (original, encoded) if value is not None):
                    raise FileToolError('PERMISSION_DENIED', '文件包含已知模型凭证，不能保存到快照')
            before = make_snapshot(original, original_mode)
            after = make_snapshot(encoded, mode if encoded is not None else None)
            if cancel_event.is_set():
                raise FileToolError('TOOL_CANCELLED', '文件补丁已取消')
            if args.operation == 'delete':
                delete_file(context.root, path, before['hash'], before['mode'])
            else:
                atomic_write(context.root, path, encoded, mode,
                             expected_exists=before['exists'], expected_hash=before['hash'],
                             expected_mode=before['mode'])
            data = {
                'path': path.relative_to(context.root.resolve()).as_posix(),
                'operation': args.operation,
                'before_sha256': before['hash'], 'after_sha256': after['hash'],
                'before_exists': before['exists'], 'after_exists': after['exists'],
                'changed': before['hash'] != after['hash'] or before['exists'] != after['exists'],
                'snapshot_ref': f'snapshots/index.json#{path.relative_to(context.root.resolve()).as_posix()}',
            }
            # 文件和 JSON 不属于同一原子事务。若实际文件已发布但索引
            # 落盘失败，结果明确保留真实副作用，不声称补丁已回滚。
            try:
                context.snapshots.record_change(path, before, after)
            except (FileToolError, OSError):
                return ToolResult(context.call_id, False, context.redactor.value(data),
                                  ToolError('SNAPSHOT_WRITE_FAILED', '文件已修改，但快照记录失败，请核对实际文件'))
            return ToolResult(context.call_id, True, context.redactor.value(data))
        except ValidationError:
            return ToolResult.fail(context.call_id, 'INVALID_ARGUMENTS', 'apply_patch 参数不符合操作约束')
        except FileToolError as exc:
            return file_failure(context, exc)
