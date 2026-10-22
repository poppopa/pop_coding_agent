"""追加日志是执行证据；恢复时闭合未知调用，绝不自动重放命令。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import fcntl
import json
import os
import uuid
from coding_agent.domain import Event
from coding_agent.security import Redactor

class SessionError(RuntimeError):
    pass

class JsonlSessionStore:
    def __init__(self, directory: Path, root: Path, session_id: str | None = None, redactor: Redactor | None = None):
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id.isalnum() or len(self.session_id) > 64:
            raise SessionError('INVALID_ARGUMENTS: 无效会话标识')
        self.root = root.resolve()
        self.directory = Path(directory) / 'sessions' / self.session_id
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.redactor = redactor or Redactor()
        self._lock = (self.directory / '.lock').open('a+')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise SessionError('SESSION_LOCKED: 此会话正在被其他进程使用') from None
        metadata = self.directory / 'metadata.json'
        if metadata.exists():
            try:
                old = json.loads(metadata.read_text())
                if old['root'] != str(self.root):
                    raise SessionError('SESSION_ROOT_MISMATCH: 会话属于另一个仓库')
            except (ValueError, KeyError) as exc:
                self.close()
                raise SessionError('SESSION_CORRUPTED: 会话元数据损坏') from exc
            except SessionError:
                self.close()
                raise
        else:
            metadata.write_text(json.dumps({'session_id': self.session_id, 'root': str(self.root), 'schema_version': 1}, ensure_ascii=False))
            metadata.chmod(0o600)
        self.path = self.directory / 'events.jsonl'
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.chmod(0o600)
        self.events = self._read()
        self.seq = self.events[-1].seq if self.events else 0

    def _read(self) -> list[Event]:
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        events = []
        offset = 0
        for index, line in enumerate(lines):
            try:
                event = Event(**json.loads(line))
                if event.seq != len(events) + 1 or event.schema_version != 1 or event.session_id != self.session_id:
                    raise ValueError('无效事件序号或版本')
                events.append(event)
                offset += len(line)
            except (ValueError, TypeError):
                if index != len(lines) - 1:
                    self.close()
                    raise SessionError('SESSION_CORRUPTED: 会话中间日志损坏') from None
                # 只修复末尾破损，原始证据副本保留；不掩盖中间的损坏。
                backup = self.path.with_suffix('.corrupted.jsonl')
                if not backup.exists():
                    backup.write_bytes(raw)
                    backup.chmod(0o600)
                self.path.write_bytes(raw[:offset])
                break
        if events and not self.path.read_bytes().endswith(b'\n'):
            with self.path.open('ab') as file:
                file.write(b'\n')
        return events

    async def append(self, event: Event) -> None:
        if event.seq != self.seq + 1:
            raise SessionError('SESSION_CORRUPTED: 追加事件序号不连续')
        safe = Event(**self.redactor.value(asdict(event)))
        with self.path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(asdict(safe), ensure_ascii=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
        self.events.append(safe)
        self.seq = safe.seq

    async def load(self, session_id: str | None = None) -> list[Event]:
        if session_id and session_id != self.session_id:
            raise SessionError('INVALID_ARGUMENTS: 会话标识不匹配')
        return list(self.events)

    def acquire_lock(self, session_id: str | None = None):
        return self._lock

    def history(self) -> list[dict]:
        """历史由已提交事件重建，pending 调用用错误结果闭合供应商协议。"""
        history: list[dict] = []
        for event in self.events:
            if event.type == 'user_message':
                history.append({'kind': 'message', 'role': 'user', 'content': event.payload['text']})
            elif event.type == 'model_response':
                history.append({'kind': 'response', 'items': deepcopy(event.payload['items'])})
            elif event.type == 'tool_finished':
                history.append({'kind': 'tool_result', 'call_id': event.payload['call_id'], 'result': deepcopy(event.payload['result'])})
            elif event.type == 'context_compacted':
                # 独立重建压缩快照，避免后续追加或调用者修改污染已提交事件。
                history = deepcopy(event.payload['history'])
        closed = []
        for index, item in enumerate(history):
            closed.append(item)
            if item.get('kind') != 'response':
                continue
            completed = set()
            for later in history[index + 1:]:
                if later.get('kind') in ('response', 'message'):
                    break
                if later.get('kind') == 'tool_result':
                    completed.add(later['call_id'])
            for output in item['items']:
                if output.get('type') == 'function_call' and output.get('call_id') not in completed:
                    call_id = output['call_id']
                    closed.append({'kind': 'tool_result', 'call_id': call_id, 'result': {
                        'call_id': call_id, 'ok': False, 'data': {},
                        'error': {'code': 'COMMAND_CANCELLED', 'message': '恢复后此调用状态未知，未自动重放', 'retryable': False},
                        'truncated': False}})
        return closed

    def close(self):
        if hasattr(self, '_lock') and not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    @staticmethod
    def list_sessions(directory: Path) -> list[dict]:
        result = []
        for file in sorted((directory / 'sessions').glob('*/metadata.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                result.append(json.loads(file.read_text()))
            except ValueError:
                continue
        return result
