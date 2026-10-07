# karen

个人 AI 工具集 monorepo（Python）。

## 包

| 包 | 说明 | 状态 |
| --- | --- | --- |
| [karen-ai](karen-ai/) | 统一 LLM API 层：消息类型、事件流、Provider/Models 注册表、鉴权、Anthropic 与 OpenAI 兼容适配器。参考 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi) 的 Python 重写。 | 核心已完成 |
| [karen-coding-agent](karen-coding-agent/) | `karen` CLI 编程助手：应用层 AgentSession（Agent + session 持久化 + 阈值/溢出自动 compaction + 自动重试 transient 失败 + hooks + 会话导航：tree/fork/clone/switch + 思考级别（变更记入 transcript，恢复/切换/派生会话时按分支恢复）+ bash 旁路（结果写回 transcript，run settle 与下次 prompt 前两次 flush）+ 会话导出 JSONL/HTML（HTML 自带渲染器、离线自包含、UTF-8 负载、只导出当前分支、渲染 bashExecution）+ 图片输入归一化（autoResize/blockImages 设置生效，读取时按当前模型解析 resize 档位））+ 交互 REPL + 无头 print/JSON 模式（最终文本或逐行 JSON 事件流）+ RPC 模式（stdin 命令 / stdout 事件的 JSON lines 协议，含 thinking-level/bash/export 命令与图片输入、prompt 的 streamingBehavior 与 busy 判定）+ 结构化系统提示词（sections：tools/rules/project_context/skills/cwd）+ 项目上下文文件（AGENTS.md/CLAUDE.md 逐级向上、SYSTEM.md/APPEND_SYSTEM.md）+ skills + 应用级工具 find/grep/ls/powershell（进程内实现 pi 的 fd/rg 语义，read 工具接图片处理管线）+ 设置文件（全局+项目双层合并 + 写入，损坏文件不覆盖）。对标 pi 的 packages/coding-agent（精简范围）。 | M9 已完成 |
| [karen-agent](karen-agent/) | 基于 karen-ai 的 agent 层：agent loop（工具执行/事件/钩子）+ Agent 类（状态/事件订阅/steering 队列/abort）+ session 持久化（JSONL、分支、fork/resume）+ 内置工具（read/write/edit/bash）+ compaction/hooks/prompt 模板（summary 请求走 assistant-call 重试）+ skills 加载器 + 执行环境（ExecutionEnv：FileSystem+Shell）+ proxy stream fn + 上下文溢出检测与恢复（overflow compaction + 一次 compact-and-retry）+ demo CLI（examples/karen_cli.py）。移植 pi 的 packages/agent 核心。 | M0–M7 已完成 |

## 布局

```
karen/
├── karen-ai/            # 最底层：统一 LLM API
│   ├── src/karen_ai/    # 包源码
│   └── tests/           # pytest 测试
├── karen-agent/         # agent 层：loop + session + 工具 + compaction
│   ├── src/karen_agent/
│   └── tests/
├── karen-coding-agent/  # 应用层：karen CLI
│   ├── src/karen_coding_agent/
│   └── tests/
└── ...
```

每个包独立打包（`pyproject.toml`），可单独安装，也可 editable 安装：

```powershell
pip install -e ./karen-ai
pip install -e ./karen-agent --no-deps          # karen-ai 为本地包
pip install -e ./karen-coding-agent --no-deps   # 提供 karen 命令
cd karen-coding-agent; pytest
```
