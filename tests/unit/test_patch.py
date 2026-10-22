"""技术方案§六个工具参数与结果：冲突保留、整批预检、创建删除、快照保护。"""
import asyncio
import hashlib
import stat
from coding_agent.tools import paths
from coding_agent.storage.snapshots import SnapshotStore

import pytest

from coding_agent.tools.patch import ApplyPatchTool
from coding_agent.tools.paths import FileToolError


async def patch(context, before, edits, path="a.py"):
    return await ApplyPatchTool().execute({"path": path, "operation": "update", "expected_sha256": hashlib.sha256(before.encode()).hexdigest(), "edits": edits}, context, asyncio.Event())


async def test_stale_fingerprint_preserves_user_modification(file_context):
    target = file_context.root / "a.py"
    target.write_text("user = 2\n")
    result = await patch(file_context, "user = 1\n", [{"old_text": "1", "new_text": "3"}])
    assert not result.ok and result.error.code == "FILE_CHANGED"
    assert target.read_text() == "user = 2\n"
    assert file_context.snapshots.diff() == ""


@pytest.mark.parametrize("before,edits", [
    pytest.param("one two", [{"old_text": "one", "new_text": "ONE"}, {"old_text": "missing", "new_text": "X"}], id="later-missing"),
    pytest.param("one one", [{"old_text": "one", "new_text": "X"}], id="duplicate"),
    pytest.param("aaa", [{"old_text": "aa", "new_text": "X"}], id="overlapping-duplicate"),
    pytest.param("abcdef", [{"old_text": "abc", "new_text": "X"}, {"old_text": "bc", "new_text": "Y"}], id="edits-overlap"),
])
async def test_invalid_edit_batch_performs_zero_writes(file_context, before, edits):
    target = file_context.root / "a.py"
    target.write_text(before)
    result = await patch(file_context, before, edits)
    assert not result.ok and result.error.code == "PATCH_NOT_UNIQUE"
    assert target.read_text() == before
    assert file_context.snapshots.diff() == ""
    assert sorted(p.name for p in file_context.root.iterdir()) == ["a.py"]


async def test_updates_use_original_content_and_preserve_permissions(file_context):
    target = file_context.root / "a.py"
    target.write_text("left middle right\n")
    target.chmod(0o751)
    result = await patch(file_context, target.read_text(), [{"old_text": "left", "new_text": "middle"}, {"old_text": "middle", "new_text": "center"}])
    assert result.ok and target.read_text() == "middle center right\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o751
    assert result.data["after_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result.data["snapshot_ref"]


async def test_create_empty_then_delete_by_fingerprint(file_context):
    tool = ApplyPatchTool()
    created = await tool.execute({"path": "a.py", "operation": "create", "content": ""}, file_context, asyncio.Event())
    assert created.ok and (file_context.root / "a.py").read_bytes() == b""
    assert created.data["before_sha256"] is None
    assert created.data["after_sha256"] == hashlib.sha256(b"").hexdigest()
    deleted = await tool.execute({"path": "a.py", "operation": "delete", "expected_sha256": created.data["after_sha256"]}, file_context, asyncio.Event())
    assert deleted.ok and not (file_context.root / "a.py").exists()
    assert deleted.data["after_exists"] is False


async def test_create_refuses_existing_file(file_context):
    (file_context.root / "a.py").write_text("user content")
    result = await ApplyPatchTool().execute({"path": "a.py", "operation": "create", "content": "bad"}, file_context, asyncio.Event())
    assert result.error.code == "FILE_CHANGED"
    assert (file_context.root / "a.py").read_text() == "user content"


@pytest.mark.parametrize("arguments", [
    pytest.param({"operation": "update", "edits": []}, id="update-missing-hash"),
    pytest.param({"operation": "create", "content": "a", "expected_sha256": "0"*64}, id="create-unrelated-hash"),
    pytest.param({"operation": "delete", "expected_sha256": "0"*64, "content": "a"}, id="delete-unrelated-content"),
])
async def test_operation_constraints_return_structured_error(file_context, arguments):
    result = await ApplyPatchTool().execute({"path": "a.py", **arguments}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS"
    assert not (file_context.root / "a.py").exists()


async def test_secret_placeholder_is_not_saved_in_snapshot(file_context):
    result = await ApplyPatchTool().execute({"path": "a.py", "operation": "create", "content": "<your-test-key>"}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "PERMISSION_DENIED"
    assert not (file_context.root / "a.py").exists()
    assert all("<your-test-key>" not in p.read_text() for p in file_context.snapshots.directory.glob("*.json"))


async def test_snapshot_restore_does_not_override_later_user_change(file_context):
    target = file_context.root / "a.py"
    target.write_text("value = 1\n")
    result = await patch(file_context, target.read_text(), [{"old_text": "1", "new_text": "2"}])
    assert result.ok
    target.write_text("value = 3\n")
    with pytest.raises(FileToolError) as error:
        file_context.snapshots.restore("a.py", result.data["after_sha256"])
    assert error.value.code == "FILE_CHANGED"
    assert target.read_text() == "value = 3\n"


async def test_snapshot_restore_preserves_user_edit_between_agent_segments(file_context):
    target = file_context.root / "a.py"
    target.write_text("value = 1\n")
    assert (await patch(file_context, target.read_text(), [{"old_text": "1", "new_text": "2"}])).ok
    target.write_text("value = 10\n")
    result = await patch(file_context, target.read_text(), [{"old_text": "10", "new_text": "11"}])
    assert result.ok
    diff = file_context.snapshots.diff()
    assert "-value = 1\n+value = 2" in diff
    assert "-value = 10\n+value = 11" in diff
    restored = file_context.snapshots.restore("a.py", result.data["after_sha256"])
    assert restored["restored"] is True
    assert target.read_text() == "value = 10\n"


async def test_atomic_publish_failure_keeps_original_and_cleans_temporary(file_context, monkeypatch):
    target = file_context.root / "a.py"
    target.write_text("value = 1\n")

    def fail_replace(source, destination):
        raise OSError("test atomic failure")

    monkeypatch.setattr(paths.os, "replace", fail_replace)
    result = await patch(file_context, target.read_text(), [{"old_text": "1", "new_text": "2"}])
    assert not result.ok and result.error is not None
    assert result.error.code and result.error.message
    assert target.read_text() == "value = 1\n"
    assert sorted(p.name for p in file_context.root.iterdir()) == ["a.py"]
    assert file_context.snapshots.diff() == ""


async def test_create_race_preserves_concurrent_user_file(file_context, monkeypatch):
    original_link = paths.os.link

    def competing_link(source, destination, **kwargs):
        destination.write_text("user created first")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(paths.os, "link", competing_link)
    result = await ApplyPatchTool().execute({"path": "a.py", "operation": "create", "content": "agent"}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "FILE_CHANGED"
    assert (file_context.root / "a.py").read_text() == "user created first"
    assert file_context.snapshots.diff() == ""


async def test_delete_conflict_keeps_user_file_and_snapshot(file_context):
    target = file_context.root / "a.py"
    target.write_text("user = 2\n")
    result = await ApplyPatchTool().execute({"path": "a.py", "operation": "delete", "expected_sha256": hashlib.sha256(b"user = 1\n").hexdigest()}, file_context, asyncio.Event())
    assert not result.ok and result.error.code == "FILE_CHANGED"
    assert target.read_text() == "user = 2\n"


async def test_snapshot_index_reopen_and_corruption_are_checked(file_context):
    target = file_context.root / "a.py"
    target.write_text("value = 1\n")
    assert (await patch(file_context, target.read_text(), [{"old_text": "1", "new_text": "2"}])).ok
    reopened = SnapshotStore(file_context.root, file_context.state_dir)
    assert reopened.diff() == file_context.snapshots.diff()
    reopened.index_path.write_text("not-json")
    with pytest.raises(FileToolError) as error:
        SnapshotStore(file_context.root, file_context.state_dir)
    assert error.value.code == "SESSION_CORRUPTED"
    assert target.read_text() == "value = 2\n"
