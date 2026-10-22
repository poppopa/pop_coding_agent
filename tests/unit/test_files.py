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
