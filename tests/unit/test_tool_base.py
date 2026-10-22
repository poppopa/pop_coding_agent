"""验收依据：技术方案§六个工具、核心契约与状态机、命令执行与授权。"""
import pytest
from pydantic import ValidationError

from coding_agent.tools.base import Tool, ToolInput, ToolRegistry


class ReadArguments(ToolInput):
    path: str


class ReadTool(Tool):
    name = "read_file"
    description = "读取测试文件"
    input_type = ReadArguments


class PatchTool(ReadTool):
    name = "apply_patch"
    mutating = True


def test_tool_arguments_accept_valid_input():
    assert ReadArguments.model_validate({"path": "a.py"}).path == "a.py"


@pytest.mark.parametrize("arguments", [
    pytest.param({"path": "a.py", "extra": True}, id="extra-field"),
    pytest.param({"path": 123}, id="wrong-type"),
    pytest.param({}, id="missing-field"),
])
def test_tool_arguments_reject_invalid_input(arguments):
    with pytest.raises(ValidationError):
        ReadArguments.model_validate(arguments)


def test_schema_preserves_tool_name_and_strict_parameters():
    schema = ReadTool().schema()
    assert schema["type"] == "function"
    assert schema["name"] == "read_file"
    assert schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["required"] == ["path"]
    assert schema["parameters"]["properties"]["path"]["type"] == "string"


def test_analysis_hides_mutating_schema_and_lookup():
    read, patch = ReadTool(), PatchTool()
    registry = ToolRegistry([read, patch])
    assert {schema["name"] for schema in registry.schemas("analysis")} == {"read_file"}
    assert registry.get("read_file", "analysis") is read
    assert registry.get("apply_patch", "analysis") is None
    assert registry.get("unknown", "execute") is None


def test_execute_exposes_both_tools():
    read, patch = ReadTool(), PatchTool()
    registry = ToolRegistry([read, patch])
    assert {schema["name"] for schema in registry.schemas("execute")} == {"read_file", "apply_patch"}
    assert registry.get("apply_patch", "execute") is patch
