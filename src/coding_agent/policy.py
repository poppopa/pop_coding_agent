"""命令授权绑定完整参数和工作目录；恢复会话时不继承授权。"""
from pathlib import Path

class Policy:
    def __init__(self, mode: str = 'execute'):
        self.mode = mode
        self._session: set[tuple] = set()
        self._once: set[tuple] = set()

    @staticmethod
    def key(argv: list[str], cwd: Path) -> tuple:
        return (tuple(argv), str(Path(cwd).resolve()))

    def approve_once(self, argv: list[str], cwd: Path):
        self._once.add(self.key(argv, cwd))

    def approve_for_session(self, argv: list[str], cwd: Path):
        self._session.add(self.key(argv, cwd))

    def evaluate(self, argv: list[str], cwd: Path) -> str:
        if self.mode == 'analysis':
            return 'deny'
        key = self.key(argv, cwd)
        if key in self._once:
            self._once.remove(key)
            return 'allow'
        return 'allow' if key in self._session else 'ask'
