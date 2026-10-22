# 个人 CLI 编程助手

一个用于个人编码和学习展示的 CLI Coding Agent。自建 Agent 内核，使用火山方舟 Ark 官方 SDK；支持文件搜索与精确补丁、命令执行、流式对话、中断和本地会话恢复。

## 安装

需要 Python 3.12，首版支持 macOS / Linux。推荐使用 uv：

```bash
uv sync
uv run cagent --help
```

也可安装为普通 Python 包：

```bash
python3.12 -m pip install .
cagent --help
```

## Ark 配置

在用户配置目录 `~/.config/cagent/config.toml` 中写非敏感配置：

```toml
model = "<YOUR_ARK_MODEL_OR_ENDPOINT_ID>"
provider = "ark"
api = "responses"
mode = "execute"
context_tokens = 32000
max_output_tokens = 4096
```

API Key 使用本机环境变量 `ARK_API_KEY`，不放在代码或配置文件中。模型标识也可以通过 `ARK_MODEL` 或命令行 `--model` 提供。模型必须支持 Responses API 和自定义函数工具；工具调用响应完整成功后才会执行。

```bash
cagent doctor
cagent
cagent --mode analysis
cagent --model '<YOUR_ARK_MODEL_OR_ENDPOINT_ID>'
```

`doctor` 只做离线检查，显示 Key 是否存在，不回显实际值，也不请求模型。缺少 Key 或模型时返回退出码 2。

## 操作方式

Enter 提交，Alt+Enter 换行；Ctrl+C 中断当前任务或清空输入，Ctrl+D 在空输入时退出。支持 `/help`、`/mode analysis|execute`、`/diff`、`/context`、`/compact`、`/exit`。

交互模式下命令提示显示完整 argv 和工作目录，可允许一次、允许本会话相同命令、拒绝。批准范围不会在恢复后自动继承。

```bash
cagent sessions
cagent --resume '<SESSION_ID>'
cagent exec '解释这个项目的目录结构' --json
cagent exec '运行项目测试' --json --allow-command '["python","-m","pytest","-q"]'
```

全局参数放在 `exec` 前。脚本模式未授权命令返回 `APPROVAL_REQUIRED` 和退出码 3，JSONL 标准输出不带终端样式。其他退出码：0 正常结束，1 任务失败，2 配置错误，130 中断；正常结束不代替实际测试结果。

## 六个工具

| 工具 | 用途 |
| --- | --- |
| read_file | 按范围读取文本，返回完整文件指纹 |
| find_files | 按模式查找文件 |
| search_text | 字面文本搜索及行号 |
| apply_patch | 通过文件指纹与唯一旧片段修改、创建或删除文件 |
| run_command | 明确 argv、超时、流式输出和进程组取消 |
| show_diff | 查看 Agent 文件修改或注明范围的 Git 差异 |

文件工具限定仓库内路径，默认跳过敏感文件与依赖目录。自由命令仍可产生外部副作用，精确命令授权不是操作系统沙箱；请在自己的可信工作目录使用。Key 过滤、输出脱敏不代替隔离。

`read_file` 默认每页最多 200 行、4096 UTF-8 字节正文，`max_bytes` 可设为 256～8192，`max_lines` 最多 400。长行会分段；未读完时，用结果的 `next_start_line`、`next_start_column` 作为下一次读取起点。列偏移从 0 开始，按脱敏后文本的 Unicode 字符计数；文件指纹始终按完整原始文件计算。模型请求会省去与正文完全相同的逐行副本，原始工具结果仍保存在会话日志中。

上下文使用 UTF-8 字节进行保守估算，`/context` 显示包含工具定义与仓库约定的请求估算及可用输入预算。达到 70% 时提前整理；不足以整理旧轮次时，只要硬预算允许就继续。超过硬预算时仍停止并给出使用量，避免发送过大的请求。

会话默认保存在 `~/.local/share/cagent/sessions/`，包含执行日志和文件快照。恢复不自动重放状态未知的命令，也不自动回滚用户改动。日志可能包含项目代码，按本地项目数据管理。

## 离线演示与学习

```bash
uv run python examples/offline_demo.py
```

演示在临时目录修复 `average([])`，使用固定模型替身，但文件读取、补丁和测试命令是真实执行。源码按域模型、内核、工具、模型适配、存储、界面划分，关键机制带中文注释。详见 `docs/架构说明.md`。

## 开发验证

```bash
uv run pytest
uv build
```

测试使用 FakeProvider 和合成 Ark 事件，不消耗真实模型调用。真实模型的多轮工具回传仍需单独进行联调。测试、演示和构建命令按实际机器资源控制并发。
