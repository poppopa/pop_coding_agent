"""技术方案§CLI、US-002/003：真实PTY输入、SIGINT、日志恢复，不连接模型。"""
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time


FIXTURE = Path(__file__).resolve().parents[1] / "cli_process_fixture.py"


def environment():
    # 只透传运行测试所需变量；不读取或转储任何凭证。
    return {"PATH": os.defpath, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"}


def prepare(tmp_path):
    (tmp_path / "config.toml").write_text('model = "test-model"\n', encoding="utf-8")


def read_until(process, descriptor, marker, timeout=8):
    data = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in data and time.monotonic() < deadline:
        ready, _, _ = select.select([descriptor], [], [], 0.05)
        if ready:
            try:
                chunk = os.read(descriptor, 65536)
            except OSError:
                break
            if not chunk:
                break
            data.extend(chunk)
            if b"\x1b[6n" in chunk:
                os.write(descriptor, b"\x1b[1;1R")
        if process.poll() is not None:
            break
    assert marker in data, f"未观察到测试标记 {marker!r}，进程退出={process.poll()}"
    return bytes(data)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def test_real_pty_multiline_sigint_and_empty_control_d(tmp_path):
    prepare(tmp_path)
    master, slave = pty.openpty()
    process = subprocess.Popen([sys.executable, str(FIXTURE), str(tmp_path), "wait"], stdin=slave, stdout=slave, stderr=slave, env=environment(), start_new_session=True)
    os.close(slave)
    try:
        read_until(process, master, b"execute >")
        os.write(master, b"first\x1b\rsecond\r")
        read_until(process, master, b"FAKE_WAIT_READY")
        history = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[0])
        assert history[-1]["content"] == "first\nsecond"
        os.kill(process.pid, signal.SIGINT)
        after_cancel = read_until(process, master, "任务已中断".encode())
        if b"execute >" not in after_cancel:
            read_until(process, master, b"execute >")
        os.write(master, b"\x04")
        assert process.wait(timeout=5) == 0
        events = [json.loads(line) for line in next((tmp_path / "state/sessions").glob("*/events.jsonl")).read_text().splitlines()]
        assert any(e["type"] == "turn_finished" and e["payload"]["status"] == "interrupted" for e in events)
    finally:
        stop(process)
        os.close(master)


def test_json_command_sigint_persists_result_and_resume_does_not_replay(tmp_path):
    prepare(tmp_path)
    process = subprocess.Popen([sys.executable, str(FIXTURE), str(tmp_path), "command"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment(), start_new_session=True)
    lines = []
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.05)
            if ready:
                line = process.stdout.readline()
                if not line:
                    break
                lines.append(line)
                row = json.loads(line)
                if row["type"] == "tool_started":
                    while not (tmp_path / "executed").exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    assert (tmp_path / "executed").read_text() == "once"
                    os.kill(process.pid, signal.SIGINT)
                    break
        remaining, errors = process.communicate(timeout=8)
        lines += remaining.splitlines(keepends=True)
        events = [json.loads(line) for line in lines]
        assert process.returncode == 130
        assert errors == b""
        assert events[-1]["type"] == "turn_finished"
        assert events[-1]["payload"]["status"] == "interrupted"
        session = next((tmp_path / "state/sessions").iterdir())
        log_before = session.joinpath("events.jsonl").read_text()
        assert "COMMAND_CANCELLED" in log_before
        resumed = subprocess.run([sys.executable, str(FIXTURE), str(tmp_path), "resume"], capture_output=True, env=environment(), timeout=8)
        assert resumed.returncode == 0
        rows = [json.loads(line) for line in resumed.stdout.splitlines()]
        assert rows[-1]["payload"]["status"] == "completed"
        assert not any(row["type"] == "tool_started" for row in rows)
        assert (tmp_path / "executed").read_text() == "once"
        requests = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
        results = [h for h in requests[-1] if h.get("kind") == "tool_result"]
        assert len(results) == 1 and results[0]["result"]["error"]["code"] == "COMMAND_CANCELLED"
    finally:
        stop(process)


def test_sigint_during_actual_approval_prompt_cancels_turn(tmp_path):
    prepare(tmp_path)
    master, slave = pty.openpty()
    process = subprocess.Popen([sys.executable, str(FIXTURE), str(tmp_path), "approval"], stdin=slave, stdout=slave, stderr=slave, env=environment(), start_new_session=True)
    os.close(slave)
    try:
        read_until(process, master, b"execute >")
        os.write(master, b"approval task\r")
        read_until(process, master, "拒绝 [n]：".encode())
        os.kill(process.pid, signal.SIGINT)
        after = read_until(process, master, "任务已中断".encode(), timeout=2)
        if b"execute >" not in after:
            read_until(process, master, b"execute >")
        os.write(master, b"\x04")
        assert process.wait(timeout=5) == 0
        assert not (tmp_path / "executed").exists()
    finally:
        stop(process)
        os.close(master)
