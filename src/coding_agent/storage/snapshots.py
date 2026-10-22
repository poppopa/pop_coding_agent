"""会话级文件快照：只记录 Agent 的文件动作，不依赖 Git 工作区基线。"""
from __future__ import annotations

import base64
import binascii
import difflib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from coding_agent.tools.paths import (
    MAX_FILE_BYTES, FileToolError, atomic_write, decode_text, delete_file,
    read_bytes, safe_path, sha256,
)


def make_snapshot(content: bytes | None, mode: int | None = None) -> dict[str, Any]:
    """base64 是恢复使用的无损来源，content 仅方便查看文本快照。

    二进制快照仍可保留字节，但文件补丁工具会单独拒绝非 UTF-8 文本。
    缺失状态的 hash 为 None，不能和空文件的 SHA-256 混为一谈。
    """
    if content is None:
        return {'exists': False, 'content': None, 'base64': None, 'hash': None, 'mode': None}
    try:
        text = decode_text(content)
    except FileToolError:
        text = None
    return {
        'exists': True, 'content': text,
        'base64': base64.b64encode(content).decode('ascii'),
        'hash': sha256(content), 'mode': mode,
    }


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes | None:
    """磁盘 JSON 也视为需要校验的输入，损坏快照绝不参与恢复。"""
    if not isinstance(snapshot, dict) or type(snapshot.get('exists')) is not bool:
        raise FileToolError('SESSION_CORRUPTED', '快照缺少有效的存在状态')
    if not snapshot['exists']:
        if any(snapshot.get(key) is not None for key in ('content', 'base64', 'hash', 'mode')):
            raise FileToolError('SESSION_CORRUPTED', '缺失文件快照含有矛盾数据')
        return None
    encoded = snapshot.get('base64')
    mode = snapshot.get('mode')
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_FILE_BYTES + 2) // 3):
        raise FileToolError('SESSION_CORRUPTED', '快照内容无效或超过大小上限')
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise FileToolError('SESSION_CORRUPTED', '快照不是有效的 base64 内容') from None
    if len(content) > MAX_FILE_BYTES or sha256(content) != snapshot.get('hash'):
        raise FileToolError('SESSION_CORRUPTED', '快照指纹与内容不一致')
    if type(mode) is not int or not 0 <= mode <= 0o7777:
        raise FileToolError('SESSION_CORRUPTED', '快照文件权限无效')
    try:
        text = decode_text(content)
    except FileToolError:
        text = None
    if snapshot.get('content') != text:
        raise FileToolError('SESSION_CORRUPTED', '快照文本与原始字节不一致')
    return content


def same_snapshot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ('exists', 'hash', 'mode'))


def unified_diff(path: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    """保留创建、删除、权限变化与缺少末尾换行标记。"""
    old = snapshot_bytes(before)
    new = snapshot_bytes(after)
    if same_snapshot(before, after):
        return ''
    lines = [f'diff --git a/{path} b/{path}\n']
    if old is None:
        lines.append(f'new file mode {stat.S_IFREG | after["mode"]:06o}\n')
    elif new is None:
        lines.append(f'deleted file mode {stat.S_IFREG | before["mode"]:06o}\n')
    elif before['mode'] != after['mode']:
        lines.extend([f'old mode {stat.S_IFREG | before["mode"]:06o}\n',
                      f'new mode {stat.S_IFREG | after["mode"]:06o}\n'])
    try:
        old_text = '' if old is None else decode_text(old)
        new_text = '' if new is None else decode_text(new)
    except FileToolError:
        lines.append('Binary files differ\n')
        return ''.join(lines)
    for line in difflib.unified_diff(
        old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        fromfile='/dev/null' if old is None else f'a/{path}',
        tofile='/dev/null' if new is None else f'b/{path}',
    ):
        lines.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
    return ''.join(lines)


class SnapshotStore:
    """可复用的会话快照索引，公开方法均为同步磁盘操作。

    同一进程内以 RLock 串行化索引读改写；跨进程的单写入者由
    SessionStore 的会话锁保证。每次操作重新加载 JSON，重建对象不会
    丢失已有变更。索引使用 0600、快照目录使用 0700，并原子替换。
    """

    def __init__(self, root: Path, state_dir: Path):
        self.root = Path(root).resolve()
        # 配置路径先规范化，兼容 macOS 的 /tmp 等系统目录链接；
        # 在规范目录内，snapshots 和 index.json 仍拒绝链接替身。
        self.state_dir = Path(state_dir).resolve()
        self.directory = self.state_dir / 'snapshots'
        self.index_path = self.directory / 'index.json'
        self._lock = threading.RLock()
        self._ensure_directory()
        self._load()

    def _ensure_directory(self) -> None:
        # 会话路径是 CLI 提供的受信配置；仍拒绝其现存父级链接，防止
        # 私有快照被无意写入链接指向的其他位置。不会 chmod 用户仓库。
        for component in (self.directory, *self.directory.parents):
            if component.is_symlink():
                raise FileToolError('SESSION_CORRUPTED', '快照目录不能经过符号链接')
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.chmod(0o700)
            if self.index_path.is_symlink():
                raise FileToolError('SESSION_CORRUPTED', '快照索引不能是符号链接')
        except PermissionError:
            raise FileToolError('PERMISSION_DENIED', '没有访问快照目录的权限') from None
        except OSError:
            raise FileToolError('SESSION_CORRUPTED', '快照目录不可用') from None

    def _load(self) -> dict[str, Any]:
        self._ensure_directory()
        try:
            metadata = self.index_path.lstat()
        except FileNotFoundError:
            return {'version': 1, 'root': str(self.root), 'changes': {}}
        except OSError:
            raise FileToolError('SESSION_CORRUPTED', '无法检查快照索引') from None
        if not stat.S_ISREG(metadata.st_mode):
            raise FileToolError('SESSION_CORRUPTED', '快照索引不是普通文件')
        try:
            descriptor = os.open(self.index_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(descriptor, 'r', encoding='utf-8') as stream:
                index = json.load(stream)
        except (OSError, UnicodeError, ValueError):
            raise FileToolError('SESSION_CORRUPTED', '快照索引无法读取或 JSON 损坏') from None
        if (not isinstance(index, dict) or index.get('version') != 1
                or index.get('root') != str(self.root) or not isinstance(index.get('changes'), dict)):
            raise FileToolError('SESSION_CORRUPTED', '快照索引版本或仓库根目录不匹配')
        for path, segments in index['changes'].items():
            if not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts:
                raise FileToolError('SESSION_CORRUPTED', '快照索引含有非法路径')
            if not isinstance(segments, list):
                raise FileToolError('SESSION_CORRUPTED', '快照变更记录无效')
            for segment in segments:
                if not isinstance(segment, dict) or set(segment) != {'before', 'after'}:
                    raise FileToolError('SESSION_CORRUPTED', '快照变更段无效')
                snapshot_bytes(segment['before'])
                snapshot_bytes(segment['after'])
        return index

    def _save(self, index: dict[str, Any]) -> None:
        self._ensure_directory()
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix='.index-', dir=self.directory)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(index, stream, ensure_ascii=True, separators=(',', ':'))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
            temporary = None
        except OSError:
            raise FileToolError('SNAPSHOT_WRITE_FAILED', '快照索引写入失败') from None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _path(self, path: str | Path) -> Path:
        resolved = safe_path(self.root, path, for_write=True)
        if resolved.is_relative_to(self.state_dir):
            raise FileToolError('PERMISSION_DENIED', '文件工具不能访问本会话的存储目录')
        return resolved

    def capture(self, path: Path) -> dict[str, Any]:
        resolved = self._path(path)
        try:
            content, mode = read_bytes(resolved)
        except FileToolError as exc:
            if exc.code == 'FILE_NOT_FOUND':
                return make_snapshot(None)
            raise
        return make_snapshot(content, mode)

    def record_change(self, path: Path, before: dict, after: dict) -> None:
        resolved = self._path(path)
        snapshot_bytes(before)
        snapshot_bytes(after)
        if same_snapshot(before, after):
            return
        relative = resolved.relative_to(self.root).as_posix()
        with self._lock:
            index = self._load()
            segments = index['changes'].setdefault(relative, [])
            # 只有连续的 Agent 变更才能合并：两次动作间若有用户编辑，
            # 新段以用户当前版本为 before。diff 因而不把用户编辑纳入
            # Agent 差异，restore 也只退回最近连续段的原版本。
            if segments and same_snapshot(segments[-1]['after'], before):
                segments[-1]['after'] = after
                if same_snapshot(segments[-1]['before'], after):
                    segments.pop()
            else:
                segments.append({'before': before, 'after': after})
            if not segments:
                index['changes'].pop(relative)
            self._save(index)

    def diff(self, path: str | None = None) -> str:
        selected = None if path is None else self._path(path).relative_to(self.root).as_posix()
        with self._lock:
            index = self._load()
            result = []
            for relative, segments in sorted(index['changes'].items()):
                if selected is not None and relative != selected:
                    continue
                self._path(relative)
                for segment in segments:
                    result.append(unified_diff(relative, segment['before'], segment['after']))
            return ''.join(result)

    def restore(self, path: str, expected_current_hash: str | None = None) -> dict:
        """只恢复最近连续 Agent 变更，绝不以调用者指纹绕过记录的 after。

        即使 expected_current_hash 为 None，也必须验证当前存在性、
        SHA-256 和权限等于修改后快照；Agent 删除的文件必须仍然缺失。
        恢复成功才移除该段，冲突和写入失败都保留索引供后续诊断。
        """
        resolved = self._path(path)
        relative = resolved.relative_to(self.root).as_posix()
        with self._lock:
            index = self._load()
            segments = index['changes'].get(relative, [])
            if not segments:
                raise FileToolError('FILE_NOT_FOUND', '该文件没有可恢复的 Agent 快照')
            segment = segments[-1]
            try:
                current = self.capture(resolved)
            except FileToolError as exc:
                if exc.code == 'PERMISSION_DENIED':
                    raise
                raise FileToolError('FILE_CHANGED', '当前文件无法匹配修改后快照，不能恢复', True) from None
            if not same_snapshot(current, segment['after']):
                raise FileToolError('FILE_CHANGED', '当前文件与 Agent 修改后指纹不一致，不能恢复', True)
            if expected_current_hash is not None and current['hash'] != expected_current_hash:
                raise FileToolError('FILE_CHANGED', '当前文件与调用者提供的指纹不一致', True)
            previous = segment['before']
            content = snapshot_bytes(previous)
            if content is None:
                if current['exists']:
                    delete_file(self.root, resolved, current['hash'], current['mode'])
            else:
                atomic_write(self.root, resolved, content, previous['mode'],
                             expected_exists=current['exists'], expected_hash=current['hash'],
                             expected_mode=current['mode'])
            segments.pop()
            if not segments:
                index['changes'].pop(relative)
            try:
                self._save(index)
            except FileToolError:
                raise FileToolError('SNAPSHOT_WRITE_FAILED', '文件已恢复，但快照索引未能保存，请核对实际文件') from None
            return {'path': relative, 'restored': True, 'before_sha256': current['hash'],
                    'after_sha256': previous['hash'], **previous}
