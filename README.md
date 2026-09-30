# karen

个人 AI 工具集 monorepo（Python）。

## 包

| 包 | 说明 | 状态 |
| --- | --- | --- |
| [karen-ai](karen-ai/) | 统一 LLM API 层：消息类型、事件流、Provider/Models 注册表、鉴权、Anthropic 与 OpenAI 兼容适配器。参考 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi) 的 Python 重写。 | 核心已完成 |
| [karen-agent](karen-agent/) | 基于 karen-ai 的 agent 层：agent loop（工具执行/事件/钩子）+ Agent 类（状态/事件订阅/steering 队列/abort）+ session 持久化（JSONL、分支、fork/resume）+ 内置工具（read/write/edit/bash）+ compaction/hooks/prompt 模板 + skills 加载器 + demo CLI（examples/karen_cli.py）。移植 pi 的 packages/agent 核心。 | M0–M5 已完成 |

## 布局

```
karen/
├── karen-ai/          # 最底层：统一 LLM API
│   ├── src/karen_ai/  # 包源码
│   └── tests/         # pytest 测试
├── karen-agent/       # agent 层：loop + session
│   ├── src/karen_agent/
│   └── tests/
└── ...
```

每个包独立打包（`pyproject.toml`），可单独安装，也可 editable 安装：

```powershell
pip install -e ./karen-ai
pip install -e ./karen-agent --no-deps  # karen-ai 为本地包
cd karen-agent; pytest
```
