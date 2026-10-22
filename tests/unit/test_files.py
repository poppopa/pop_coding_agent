"""技术方案§六个工具的参数与结果：行范围、完整指纹、默认边界与忽略规则。"""
import asyncio
import hashlib
from pathlib import Path

import pytest

from coding_agent.tools import files
from coding_agent.tools.files import FindFilesTool, ReadFileTool, SearchTextTool


async def test_read_defaults_two_hundred_lines_with_full_file_hash(file_context):
    content = "".join(f"第{i}行\n" for i in range(1, 251)).encode()
    (file_context.root / "a.py").write_bytes(content)
    result = await ReadFileTool().execute({"path": "a.py"}, file_context, asyncio.Event())
    assert result.ok and result.truncated
    assert result.data["text"] == "".join(f"第{i}行\n" for i in range(1, 201))
    assert (result.data["start_line"], result.data["end_line"], result.data["total_lines"]) == (1, 200, 250)
    assert result.data["sha256"] == hashlib.sha256(content).hexdigest()
    assert [line["line_number"] for line in result.data["lines"]] == list(range(1, 201))


async def test_read_start_line_and_four_hundred_limit(file_context):
    (file_context.root / "a.py").write_text("".join(f"{i}\n" for i in range(1, 402)))
    result = await ReadFileTool().execute({"path": "a.py", "start_line": 2, "max_lines": 400}, file_context, asyncio.Event())
    assert result.ok
    assert result.data["text"] == "".join(f"{i}\n" for i in range(2, 402))
    invalid = await ReadFileTool().execute({"path": "a.py", "max_lines": 401}, file_context, asyncio.Event())
    assert invalid.error.code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("content", [pytest.param(b"\xff", id="non-utf8"), pytest.param(b"a\x00b", id="binary"), pytest.param(b"a" * (1024**2 + 1), id="over-one-mib")])
async def test_read_rejects_unsupported_files(file_context, content):
    (file_context.root / "a.py").write_bytes(content)
    result = await ReadFileTool().execute({"path": "a.py"}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "UNSUPPORTED_FILE"


async def test_one_mib_utf8_file_is_accepted(file_context):
    (file_context.root / "a.py").write_bytes(b"a" * 1024**2)
    result = await ReadFileTool().execute({"path": "a.py"}, file_context, asyncio.Event())
    assert result.ok and result.data["size_bytes"] == 1024**2


async def test_symlink_and_parent_traversal_cannot_escape_root(file_context, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("private")
    (file_context.root / "alias.py").symlink_to(outside)
    for path in ("alias.py", "../outside.py"):
        result = await ReadFileTool().execute({"path": path}, file_context, asyncio.Event())
        assert not result.ok and result.error.code == "PATH_OUTSIDE_ROOT"


@pytest.mark.parametrize("fallback", [False, True], ids=["rg", "python"])
async def test_find_search_ignore_sensitive_files_and_search_literal(file_context, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(files.shutil, "which", lambda name: None)
    root = file_context.root
    (root / "visible.py").write_text("literal .* match\nregex should not match\n")
    for name in (".env", "private.key"):
        (root / name).write_text("literal .* match")
    for directory in (".git", ".venv", "node_modules", "dist"):
        (root / directory).mkdir()
        (root / directory / "hidden.py").write_text("literal .* match")
    found = await FindFilesTool().execute({"pattern": "**/*"}, file_context, asyncio.Event())
    assert found.ok and set(found.data["paths"]) == {"visible.py"}
    result = await SearchTextTool().execute({"query": ".*"}, file_context, asyncio.Event())
    assert result.ok and result.data["count"] == 1
    match = result.data["matches"][0]
    assert (match["path"], match["line_number"], match["text"]) == ("visible.py", 1, "literal .* match")


async def test_find_search_default_hundred_and_max_five_hundred(file_context):
    for i in range(501):
        (file_context.root / f"{i}.py").write_text("needle\n")
    for tool, arguments, field in [(FindFilesTool(), {"pattern": "*.py"}, "paths"), (SearchTextTool(), {"query": "needle"}, "matches")]:
        default = await tool.execute(arguments, file_context, asyncio.Event())
        assert default.ok and len(default.data[field]) == 100 and default.truncated
        maximum = await tool.execute({**arguments, "max_results": 500}, file_context, asyncio.Event())
        assert maximum.ok and len(maximum.data[field]) == 500 and maximum.truncated
        invalid = await tool.execute({**arguments, "max_results": 501}, file_context, asyncio.Event())
        assert invalid.error.code == "INVALID_ARGUMENTS"


async def test_known_placeholder_is_redacted_in_read_results(file_context):
    content = "text <your-test-key>\n"
    (file_context.root / "a.py").write_text(content)
    result = await ReadFileTool().execute({"path": "a.py"}, file_context, asyncio.Event())
    assert result.ok and result.data["text"] == "text <REDACTED>\n"
    assert result.data["sha256"] == hashlib.sha256(content.encode()).hexdigest()


# 本轮新增依据：CLI 编程助手上下文修复契约§文件读取行为。
async def read_all_pages(file_context, content, *, expected=None, max_bytes=256, max_lines=200):
    (file_context.root / "paged.py").write_bytes(content)
    expected = content.decode("utf-8") if expected is None else expected
    expected_lines = expected.splitlines(keepends=True)
    cursor, seen, pieces, pages = (1, 0), set(), [], []
    for _ in range(500):
        assert cursor not in seen, "续读游标必须前进，不能重复同一页"
        seen.add(cursor)
        result = await ReadFileTool().execute({"path": "paged.py", "start_line": cursor[0],
            "start_column": cursor[1], "max_bytes": max_bytes, "max_lines": max_lines}, file_context, asyncio.Event())
        assert result.ok and result.error is None
        data = result.data
        assert result.call_id == file_context.call_id
        assert data["sha256"] == hashlib.sha256(content).hexdigest()
        assert data["size_bytes"] == len(content) and data["total_lines"] == len(expected_lines)
        assert (data["start_line"], data["start_column"]) == cursor
        assert data["truncated"] == result.truncated
        assert len(data["text"].encode("utf-8")) <= max_bytes
        assert "".join(line["text"] for line in data["lines"]) == data["text"]
        assert len(data["lines"]) <= max_lines
        assert [line["line_number"] for line in data["lines"]] == list(range(cursor[0], cursor[0] + len(data["lines"])))
        assert data["end_line"] == (data["lines"][-1]["line_number"] if data["lines"] else None)
        begin = sum(len(line) for line in expected_lines[:cursor[0] - 1]) + cursor[1]
        assert data["text"] == expected[begin:begin + len(data["text"])]
        pieces.append(data["text"])
        pages.append(data)
        following = (data["next_start_line"], data["next_start_column"])
        if following == (None, None):
            assert "".join(pieces) == expected
            return pages
        assert all(isinstance(part, int) for part in following)
        next_offset = sum(len(line) for line in expected_lines[:following[0] - 1]) + following[1]
        assert next_offset == begin + len(data["text"]), "游标必须指向脱敏后下一字符"
        assert data["text"] and result.truncated
        cursor = following
    pytest.fail("有限文件分页应在限制次数内读完")


async def test_default_byte_page_bounds_a_one_mib_single_line(file_context):
    content = b"x" * 1024**2
    (file_context.root / "large.py").write_bytes(content)
    result = await ReadFileTool().execute({"path": "large.py"}, file_context, asyncio.Event())
    assert result.ok and result.truncated
    assert result.data["text"] == "x" * 4096
    assert result.data["lines"] == [{"line_number": 1, "text": "x" * 4096}]
    assert (result.data["next_start_line"], result.data["next_start_column"]) == (1, 4096)
    assert result.data["sha256"] == hashlib.sha256(content).hexdigest()
    assert result.data["size_bytes"] == len(content)


@pytest.mark.parametrize("content,max_lines", [
    pytest.param(("中😀文é" * 180).encode(), 200, id="multibyte-long-line"),
    pytest.param(b"first\r\n" + b"x" * 246 + b"\r\nlast\r\n", 200, id="split-crlf"),
    pytest.param(("甲\n" * 9 + "末行").encode(), 2, id="line-limit"),
    pytest.param(b"x" * 256, 200, id="exact-byte-eof"),
    pytest.param(b"", 200, id="empty-file"),
], ids=["multibyte-long-line", "split-crlf", "line-limit", "exact-byte-eof", "empty-file"])
async def test_returned_cursors_reconstruct_all_utf8_without_gaps_or_duplicates(file_context, content, max_lines):
    pages = await read_all_pages(file_context, content, max_lines=max_lines)
    assert pages[-1]["next_start_line"] is None and pages[-1]["next_start_column"] is None
    if content == b"":
        assert pages[0]["text"] == "" and pages[0]["lines"] == [] and pages[0]["end_line"] is None
        assert pages[0]["truncated"] is False


@pytest.mark.parametrize("max_bytes", [256, 8192], ids=["minimum-bytes", "maximum-bytes"])
async def test_allowed_byte_limits_are_enforced_and_continue(file_context, max_bytes):
    content = b"z" * (max_bytes + 1)
    pages = await read_all_pages(file_context, content, max_bytes=max_bytes)
    assert pages[0]["text"] == "z" * max_bytes
    assert len(pages) == 2 and pages[1]["text"] == "z"


@pytest.mark.parametrize("arguments", [
    pytest.param({"max_bytes": 255}, id="below-min-bytes"),
    pytest.param({"max_bytes": 8193}, id="above-max-bytes"),
    pytest.param({"max_lines": 0}, id="below-min-lines"),
    pytest.param({"max_lines": 401}, id="above-max-lines"),
    pytest.param({"start_line": 0}, id="zero-line"),
    pytest.param({"start_column": -1}, id="negative-column"),
    pytest.param({"start_column": "1"}, id="noninteger-column"),
], ids=["below-min-bytes", "above-max-bytes", "below-min-lines", "above-max-lines", "zero-line", "negative-column", "noninteger-column"])
async def test_invalid_paging_arguments_return_structured_error(file_context, arguments):
    (file_context.root / "a.py").write_text("safe\n")
    result = await ReadFileTool().execute({"path": "a.py", **arguments}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS"
    assert result.call_id == file_context.call_id and result.error.retryable is False


async def test_start_column_uses_unicode_characters_and_accepts_line_end(file_context):
    (file_context.root / "a.py").write_bytes("甲😀乙\r\n下一行".encode())
    part = await ReadFileTool().execute({"path": "a.py", "start_column": 2}, file_context, asyncio.Event())
    assert part.ok and part.data["text"] == "乙\r\n下一行"
    assert (part.data["start_line"], part.data["start_column"], part.data["end_line"]) == (1, 2, 2)
    line_end = await ReadFileTool().execute({"path": "a.py", "start_column": 5}, file_context, asyncio.Event())
    assert line_end.ok and line_end.data["text"] == "下一行"
    assert (line_end.data["start_line"], line_end.data["start_column"]) == (2, 0)
    invalid = await ReadFileTool().execute({"path": "a.py", "start_column": 6}, file_context, asyncio.Event())
    assert not invalid.ok and invalid.error.code == "INVALID_ARGUMENTS"


async def test_beyond_eof_only_accepts_zero_column(file_context):
    (file_context.root / "a.py").write_text("one\ntwo\n")
    empty = await ReadFileTool().execute({"path": "a.py", "start_line": 3}, file_context, asyncio.Event())
    assert empty.ok and empty.data["text"] == "" and empty.data["lines"] == []
    assert empty.data["start_line"] == 3 and empty.data["end_line"] is None
    assert empty.data["next_start_line"] is None and empty.data["next_start_column"] is None
    invalid = await ReadFileTool().execute({"path": "a.py", "start_line": 3, "start_column": 1}, file_context, asyncio.Event())
    assert not invalid.ok and invalid.error.code == "INVALID_ARGUMENTS"


async def test_known_placeholder_crossing_page_is_redacted_before_paging(file_context):
    prefix = "中" * 82 + "abc"
    content = (prefix + "<your-test-key>" + "尾" * 100 + "\r\n").encode()
    expected = prefix + "<REDACTED>" + "尾" * 100 + "\r\n"
    pages = await read_all_pages(file_context, content, expected=expected)
    assert "<your-test-key>" not in "".join(page["text"] for page in pages)
    assert "".join(page["text"] for page in pages).count("<REDACTED>") == 1


async def test_pattern_redaction_at_page_boundary_preserves_budget_and_cursor(file_context):
    # 全部是合成占位符；被替换标记恰好跨页，不能再次脱敏改变页长度。
    prefix = "a" * 246 + "\napi_key="
    content = (prefix + "<your-test-key>" + "\n尾行\n").encode()
    expected = prefix + "<REDACTED>\n尾行\n"
    await read_all_pages(file_context, content, expected=expected)


async def test_missing_sensitive_and_cancelled_paged_reads_keep_error_identity(file_context):
    (file_context.root / "a.py").write_text("safe\n")
    missing = await ReadFileTool().execute({"path": "missing.py"}, file_context, asyncio.Event())
    assert not missing.ok and missing.error.code == "FILE_NOT_FOUND"
    sensitive = await ReadFileTool().execute({"path": ".env"}, file_context, asyncio.Event())
    # 沿用test_patch.py:test_patch_rejects_sensitive_paths的既有权限错误身份。
    assert not sensitive.ok and sensitive.error.code == "PERMISSION_DENIED"
    cancel = asyncio.Event()
    cancel.set()
    cancelled = await ReadFileTool().execute({"path": "a.py"}, file_context, cancel)
    assert not cancelled.ok and cancelled.error.code == "TOOL_CANCELLED"
