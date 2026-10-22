"""固定离线进程夹具：真实CLI、终端和持久化，模型只产生合成响应。"""
import asyncio
import json
from pathlib import Path
import sys

from coding_agent.cli import run_cli
from coding_agent.domain import ModelFrame, ToolCall


class Provider:
    def __init__(self, mode, directory):
        self.mode = mode
        self.directory = directory

    async def stream(self, request, cancel_event):
        with (self.directory / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(list(request.history), ensure_ascii=False) + "\n")
        if self.mode == "wait":
            yield ModelFrame("text_delta", text="FAKE_WAIT_READY\n")
            await asyncio.Event().wait()
        elif self.mode in ("command", "approval"):
            script = "from pathlib import Path; import time; Path('executed').write_text('once'); print('COMMAND_WAIT_READY',flush=True); time.sleep(60)"
            call = ToolCall("request-1", "call-1", "run_command", json.dumps({"argv": [sys.executable, "-c", script]}))
            yield ModelFrame("response_finished", status="completed", calls=(call,))
        else:
            yield ModelFrame("text_delta", text="恢复完成\n")
            yield ModelFrame("response_finished", status="completed")

    async def close(self):
        pass


directory = Path(sys.argv[1])
mode = sys.argv[2]
argv = ["--config", str(directory / "config.toml"), "--state-dir", str(directory / "state"), "--root", str(directory)]
if mode == "resume":
    session = next((directory / "state/sessions").iterdir()).name
    argv += ["--resume", session, "exec", "继续说明", "--json"]
elif mode == "command":
    script = "from pathlib import Path; import time; Path('executed').write_text('once'); print('COMMAND_WAIT_READY',flush=True); time.sleep(60)"
    argv += ["exec", "运行固定命令", "--json", "--allow-command", json.dumps([sys.executable, "-c", script])]
elif mode not in ("wait", "approval"):
    raise ValueError("未知测试模式")
raise SystemExit(run_cli(argv, {"provider": Provider(mode, directory)}))
