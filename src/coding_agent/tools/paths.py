"""文件工具共用的路径边界、有限读取和原子修改机制。"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

MAX_FILE_BYTES = 1024 * 1024
IGNORED_DIRECTORIES = frozenset({
    '.git', '.hg', '.svn', '.ssh', '.aws', '.azure', '.gnupg',
    'node_modules', 'vendor', '.venv', 'venv', 'env', '__pycache__',
    '.tox', '.nox', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    'dist', 'build', 'target', '.next', '.nuxt', '.output', 'coverage',
    '.coding_agent', '.coding-agent',
})
SENSITIVE_PATTERNS = (
    '.env', '.env.*', '*.pem', '*.key', '*.p12', '*.pfx', '*.keystore',
    'id_rsa*', 'id_dsa*', 'id_ecdsa*', 'id_ed25519*',
    'credentials', 'credentials.*', 'secrets', 'secrets.*',
    '.netrc', '.npmrc', '.pypirc',
)


class FileToolError(Exception):
    """预期的文件错误携带稳定编号，不把系统异常全文送进模型上下文。"""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def is_ignored(relative: Path) -> bool:
    return any(
        part.lower() in IGNORED_DIRECTORIES
        or any(fnmatch.fnmatchcase(part.lower(), pattern) for pattern in SENSITIVE_PATTERNS)
        for part in relative.parts
    )


def safe_path(root: Path, path: str | Path, *, for_write: bool = False) -> Path:
    """同时检查输入路径和实际路径，避免链接别名绕过敏感目录规则。

    resolve(strict=False) 支持尚未创建的文件，但其现存父目录仍会真实解析。
    不使用字符串前缀判断目录归属：/repo-other 不是 /repo 的子目录。
    写操作拒绝路径链上的全部符号链接；否则 replace 会替换链接本身，
    而 read_file 却读取链接目标，导致指纹、快照和实际修改对象不一致。
    """
    root = Path(root).resolve()
    raw = os.fspath(path)
    if not raw or '\x00' in raw:
        raise FileToolError('INVALID_ARGUMENTS', '文件路径必须非空且不能包含空字符')
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        lexical = candidate.relative_to(root)
    except ValueError:
        raise FileToolError('PATH_OUTSIDE_ROOT', '路径不在仓库根目录内') from None
    if is_ignored(lexical):
        raise FileToolError('PERMISSION_DENIED', '路径属于默认忽略的敏感文件或目录')
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        raise FileToolError('UNSUPPORTED_FILE', '路径无法解析或存在符号链接循环') from None
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise FileToolError('PATH_OUTSIDE_ROOT', '实际路径或符号链接目标超出仓库根目录') from None
    if not relative.parts:
        raise FileToolError('UNSUPPORTED_FILE', '工具需要文件路径，不能使用仓库根目录')
    if is_ignored(relative):
        raise FileToolError('PERMISSION_DENIED', '实际路径属于默认忽略的敏感文件或目录')
    if for_write:
        current = root
        for part in lexical.parts:
            current = current / part
            if current.is_symlink():
                raise FileToolError('UNSUPPORTED_FILE', '写入和恢复不支持符号链接路径')
    return resolved


def matches_glob(relative: str, pattern: str) -> bool:
    """无斜杠的模式匹配文件名；**/ 也匹配仓库根层的文件。"""
    if '/' not in pattern:
        return fnmatch.fnmatchcase(Path(relative).name, pattern)
    if fnmatch.fnmatchcase(relative, pattern):
        return True
    return pattern.startswith('**/') and matches_glob(relative, pattern[3:])


def iter_files(root: Path, pattern: str = '**/*', *, excluded: tuple[Path, ...] = ()) -> Iterator[Path]:
    """确定顺序地遍历普通文件；不进入链接目录，因此不会产生目录环。

    不依赖 rg 的平台配置，查找和搜索使用同一套 Python 回退规则。
    单个无法访问的目录不会阻断其他文件，直接访问该文件则返回明确错误。
    """
    root = Path(root).resolve()
    excluded = tuple(Path(p).resolve() for p in excluded)
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if not is_ignored((parent / name).relative_to(root))
            and not (parent / name).is_symlink()
            and not any((parent / name).is_relative_to(p) for p in excluded)
        )
        for name in sorted(files):
            candidate = parent / name
            relative = candidate.relative_to(root)
            if is_ignored(relative) or not matches_glob(relative.as_posix(), pattern):
                continue
            try:
                resolved = safe_path(root, candidate)
                if not resolved.is_file() or any(resolved.is_relative_to(p) for p in excluded):
                    continue
            except (FileToolError, OSError):
                continue
            # 保留调用者看到的链接别名，但每次读取仍重新检查实际目标。
            yield candidate


def read_bytes(path: Path) -> tuple[bytes, int]:
    """最多读取 1 MiB + 1 字节，特殊文件在读取前拒绝，避免 FIFO 阻塞。

    最后一段路径以 O_NOFOLLOW 打开；打开前后检查 inode、大小和时间，
    防止常规编辑在读取过程中让内容与权限来自不同版本。该检查不是
    面向恶意并发进程的文件系统沙箱，受信任工作区仍是首版前提。
    """
    descriptor = None
    try:
        initial = path.lstat()
        if not stat.S_ISREG(initial.st_mode):
            raise FileToolError('UNSUPPORTED_FILE', '只支持普通文件')
        if initial.st_size > MAX_FILE_BYTES:
            raise FileToolError('UNSUPPORTED_FILE', '文件超过 1 MiB 上限')
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FileToolError('UNSUPPORTED_FILE', '只支持普通文件')
        with os.fdopen(descriptor, 'rb') as stream:
            descriptor = None
            content = stream.read(MAX_FILE_BYTES + 1)
            final = os.fstat(stream.fileno())
        if len(content) > MAX_FILE_BYTES:
            raise FileToolError('UNSUPPORTED_FILE', '文件超过 1 MiB 上限')
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if signature(initial) != signature(opened) or signature(opened) != signature(final):
            raise FileToolError('FILE_CHANGED', '文件在读取期间发生变化，请重新读取', True)
        return content, stat.S_IMODE(final.st_mode)
    except FileNotFoundError:
        raise FileToolError('FILE_NOT_FOUND', '文件不存在') from None
    except PermissionError:
        raise FileToolError('PERMISSION_DENIED', '没有读取文件的权限') from None
    except OSError:
        raise FileToolError('UNSUPPORTED_FILE', '无法安全读取文件') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def decode_text(content: bytes) -> str:
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        raise FileToolError('UNSUPPORTED_FILE', '文件不是有效的 UTF-8 文本') from None
    if '\x00' in text:
        raise FileToolError('UNSUPPORTED_FILE', '不支持二进制文件')
    return text


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_current(root: Path, path: Path, *, exists: bool, expected_hash: str | None,
                  expected_mode: int | None = None) -> None:
    """副作用前重新解析路径并核对内容，缺失文件也有独立的存在性约束。"""
    checked = safe_path(root, path, for_write=True)
    if checked != path:
        raise FileToolError('FILE_CHANGED', '文件实际路径发生变化', True)
    if not exists:
        if os.path.lexists(path):
            raise FileToolError('FILE_CHANGED', '创建目标已被其他编辑者写入', True)
        return
    try:
        content, mode = read_bytes(path)
    except FileToolError as exc:
        if exc.code == 'FILE_NOT_FOUND':
            raise FileToolError('FILE_CHANGED', '文件已被其他编辑者删除', True) from None
        raise
    if sha256(content) != expected_hash or (expected_mode is not None and mode != expected_mode):
        raise FileToolError('FILE_CHANGED', '文件内容或权限已改变，请重新读取后重试', True)


def atomic_write(root: Path, path: Path, content: bytes, mode: int, *,
                 expected_exists: bool, expected_hash: str | None,
                 expected_mode: int | None = None) -> None:
    """完整临时文件刷盘后再发布，失败不会留下半份目标文件。

    更新通过同目录 os.replace 原子替换；创建使用硬链接的排他发布，
    避免检查目标不存在之后直接 replace 覆盖恰好出现的用户新文件。
    临时文件不跨文件系统，且写入前设置目标权限，保留可执行位。
    """
    if len(content) > MAX_FILE_BYTES:
        raise FileToolError('UNSUPPORTED_FILE', '修改后的文件超过 1 MiB 上限')
    safe_path(root, path, for_write=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix='.cagent-', dir=path.parent)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        check_current(root, path, exists=expected_exists, expected_hash=expected_hash,
                      expected_mode=expected_mode)
        if expected_exists:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                raise FileToolError('FILE_CHANGED', '创建目标已存在，未覆盖该文件', True) from None
    except FileNotFoundError:
        raise FileToolError('FILE_NOT_FOUND', '目标文件的父目录不存在') from None
    except PermissionError:
        raise FileToolError('PERMISSION_DENIED', '没有修改目标文件的权限') from None
    except OSError:
        raise FileToolError('FILE_WRITE_FAILED', '原子写入失败') from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def delete_file(root: Path, path: Path, expected_hash: str, expected_mode: int | None = None) -> None:
    check_current(root, path, exists=True, expected_hash=expected_hash, expected_mode=expected_mode)
    try:
        path.unlink()
    except FileNotFoundError:
        raise FileToolError('FILE_CHANGED', '文件已被其他编辑者删除', True) from None
    except PermissionError:
        raise FileToolError('PERMISSION_DENIED', '没有删除文件的权限') from None
    except OSError:
        raise FileToolError('FILE_WRITE_FAILED', '删除文件失败') from None
