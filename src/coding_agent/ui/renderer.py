"""交互式展示和 JSONL 都消费相同事件，脚本输出不带终端装饰。"""
from dataclasses import asdict
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

class JsonRenderer:
    def handle(self, event):
        print(json.dumps(asdict(event), ensure_ascii=False), flush=True)

class TerminalRenderer:
    def __init__(self, console=None):
        self.console = console or Console()
        self.streaming = False

    def handle(self, event):
        p = event.payload
        if event.type == 'text_delta':
            self.console.print(p['text'], end='', markup=False, highlight=False)
            self.streaming = True
        elif event.type == 'state_changed' and p.get('state') == 'context_ready':
            self._break()
            self.console.print('模型摘要不可用，已按本地执行事实整理，继续任务。' if p.get('source') == 'local'
                               else '上下文已整理，继续任务。', markup=False, highlight=False)
        elif event.type == 'tool_started':
            self._break()
            self.console.print(f"[cyan]工具[/cyan] {p['name']}", markup=True)
        elif event.type == 'tool_output':
            self.console.print(p['text'], end='', markup=False, highlight=False)
        elif event.type == 'tool_finished':
            self._break()
            result = p['result']
            if not result['ok']:
                error = result.get('error') or {}
                self.console.print(f"{error.get('code', 'ERROR')}: {error.get('message', '')}", style='yellow', markup=False)
            if p['name'] == 'show_diff' and result['ok']:
                self.console.print(Syntax(result.get('data', {}).get('diff', '') or '没有修改', 'diff'))
            if p['name'] == 'run_command':
                self.console.print(f"命令退出码：{result.get('data', {}).get('exit_code', '未执行')}", markup=False)
        elif event.type == 'turn_finished':
            self._break()
            if p.get('error'):
                self.console.print(p['error'].get('message', ''), style='red', markup=False)
            if p.get('changes'):
                self.console.print('修改文件：' + '、'.join(change['path'] for change in p['changes']), markup=False)
            if p.get('verifications'):
                checks = p['verifications']
                self.console.print(f"实际验证：{sum(check['passed'] for check in checks)}/{len(checks)} 条命令通过", markup=False)
            elif p.get('changes'):
                self.console.print('本轮未执行验证命令。', style='yellow')
            if p['status'] == 'interrupted':
                self.console.print('任务已中断，已发生的修改保留。', style='yellow')

    def _break(self):
        if self.streaming:
            self.console.print()
            self.streaming = False
