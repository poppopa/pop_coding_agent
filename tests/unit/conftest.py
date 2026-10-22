"""文件工具使用临时仓库与分离存储，禁止真实模型凭证进入测试。"""
import pytest
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.storage.snapshots import SnapshotStore
from coding_agent.tools.base import ToolContext


@pytest.fixture
def file_context(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    state = tmp_path / "state"
    snapshots = SnapshotStore(root, state)

    async def emit(kind, payload):
        pass

    return ToolContext(root, "test-call", snapshots, Policy(), emit, Redactor(["<your-test-key>"]), state)
