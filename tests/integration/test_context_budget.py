"""修复契约§用户可观测判据：真实CLI入口、六工具装配、会话日志与四文件回传。"""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json

import pytest

from coding_agent import cli
from coding_agent.config import AgentConfig
from coding_agent.domain import ModelFrame, ToolCall
from coding_agent.storage.session import JsonlSessionStore
from coding_agent.ui import terminal


@pytest.mark.parametrize("entry", ["exec", "interactive"], ids=["exec-json", "interactive"])
async def test_four_files_in_one_response_finish_with_default_tools_and_original_log(tmp_path, monkeypatch, capsys, entry):
    # 外部依赖只有模型替身和终端输入；业务内核、ContextManager、read_file、日志全部真实。
    for name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    root, state = tmp_path / "repo", tmp_path / "state"
    root.mkdir()
    contents = {}
    for number in range(4):
        content = (chr(97 + number) * 10000).encode()
        contents[f"module_{number}.py"] = content
        (root / f"module_{number}.py").write_bytes(content)
    # 用用户项目规则参与真实预算，不能只测省略schemas的宽松请求。
    (root / "AGENTS.md").write_text("遵守原用户目标，只解释项目。\n" + "规则正文" * 100)
    prompt = "解释一下当前这个项目，连续读取四份源码后给出项目解释"
    calls = tuple(ToolCall("request-four-files", f"file-{number}", "read_file",
        json.dumps({"path": f"module_{number}.py"})) for number in range(4))
    raw_output = tuple({"type": "function_call", "call_id": call.call_id, "name": call.name,
        "arguments": call.arguments_json, "provider_metadata": {"keep": number}} for number, call in enumerate(calls))

    class FakeProvider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            await asyncio.sleep(0)
            if len(self.requests) == 1:
                yield ModelFrame("response_finished", status="completed", calls=calls, provider_output=raw_output)
            else:
                assert len(self.requests) == 2, "不能额外发摘要请求或重复模型请求"
                yield ModelFrame("text_delta", text="四份源码读取完成，项目解释已完成。")
                yield ModelFrame("response_finished", status="completed")

    if entry == "interactive":
        prompts = iter([prompt, "/exit"])

        class Prompt:
            def __init__(self, **kwargs):
                pass

            async def prompt_async(self, text):
                return next(prompts)

        monkeypatch.setattr(terminal, "PromptSession", Prompt)
        monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    args = cli.parser().parse_args(["--root", str(root)] + (["exec", prompt, "--json"] if entry == "exec" else []))
    provider = FakeProvider()
    config = AgentConfig(model="test-model", state_dir=state)
    code = await cli.run_async(args, config, {"provider": provider})
    captured = capsys.readouterr()
    assert code == 0
    assert "四份源码读取完成" in captured.out
    assert "CONTEXT_LIMIT" not in captured.out + captured.err
    assert len(provider.requests) == 2
    for request in provider.requests:
        assert {schema["name"] for schema in request.tool_schemas} == {
            "read_file", "find_files", "search_text", "apply_patch", "run_command", "show_diff"}
        used = len(json.dumps([request.instructions, request.history, request.tool_schemas], ensure_ascii=False).encode()) + 512
        assert used <= 32000 - 4096 - 2048
        assert request.max_output_tokens == 4096
    second = provider.requests[1]
    # 此触发输入确实跨过软阈值，且硬预算足够；不会误把“无需整理”当任务失败。
    second_used = len(json.dumps([second.instructions, second.history, second.tool_schemas], ensure_ascii=False).encode()) + 512
    assert second_used >= 32000 * .7
    assert second.history[0] == {"kind": "message", "role": "user", "content": prompt}
    assert [entry["items"] for entry in second.history if entry.get("kind") == "response"] == [list(raw_output)]
    returned = [entry for entry in second.history if entry.get("kind") == "tool_result"]
    assert [entry["call_id"] for entry in returned] == [call.call_id for call in calls]
    for number, entry in enumerate(returned):
        result = entry["result"]
        data = result["data"]
        assert result["call_id"] == calls[number].call_id and result["ok"] and result["error"] is None
        assert "lines" not in data
        assert data["text"] == chr(97 + number) * 4096
        assert data["sha256"] == hashlib.sha256(contents[f"module_{number}.py"]).hexdigest()
        assert data["size_bytes"] == 10000 and result["truncated"]
        assert (data["next_start_line"], data["next_start_column"]) == (1, 4096)
    directory = next((state / "sessions").iterdir())
    original_log = (directory / "events.jsonl").read_bytes()
    rows = [json.loads(line) for line in original_log.splitlines()]
    assert rows[-1]["type"] == "turn_finished" and rows[-1]["payload"]["status"] == "completed"
    assert not any(row["type"] == "context_compacted" for row in rows)
    stored_results = [row["payload"] for row in rows if row["type"] == "tool_finished"]
    assert len(stored_results) == 4
    for stored, projected in zip(stored_results, returned):
        expected = deepcopy(stored["result"])
        assert expected["data"].pop("lines") == [{"line_number": 1, "text": expected["data"]["text"]}]
        assert projected["result"] == expected
        assert stored["call_id"] == projected["call_id"]
    restored = JsonlSessionStore(state, root, directory.name)
    try:
        history = restored.history()
        assert [entry["result"] for entry in history if entry.get("kind") == "tool_result"] == [item["result"] for item in stored_results]
        assert restored.path.read_bytes() == original_log
    finally:
        restored.close()
    if entry == "exec":
        emitted = [json.loads(line) for line in captured.out.splitlines()]
        assert emitted == rows
