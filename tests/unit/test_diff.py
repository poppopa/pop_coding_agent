"""技术方案§六个工具：Agent差异仅含自身修改，Git差异明确标记用户修改。"""
import asyncio
import subprocess

from coding_agent.tools.diff import ShowDiffTool
from coding_agent.tools.patch import ApplyPatchTool


async def test_agent_diff_marks_create_and_excludes_existing_user_file(file_context):
    (file_context.root / "user.py").write_text("user content\n")
    result = await ApplyPatchTool().execute({"path": "a.py", "operation": "create", "content": "agent = 1\n"}, file_context, asyncio.Event())
    assert result.ok
    diff = await ShowDiffTool().execute({}, file_context, asyncio.Event())
    assert diff.ok and diff.data["scope"] == "agent"
    assert diff.data["includes_user_changes"] is False
    assert "new file mode" in diff.data["diff"]
    assert "agent = 1" in diff.data["diff"]
    assert "user content" not in diff.data["diff"]


async def test_git_diff_marks_user_changes_without_touching_worktree(file_context):
    subprocess.run(["git", "init", "-q"], cwd=file_context.root, check=True)
    target = file_context.root / "user.py"
    target.write_text("user content\n")
    result = await ShowDiffTool().execute({"scope": "git"}, file_context, asyncio.Event())
    assert result.ok and result.data["includes_user_changes"] is True
    assert "用户" in result.data["note"]
    assert "+user content" in result.data["diff"]
    assert target.read_text() == "user content\n"
