# karen

个人 AI 工具集 monorepo（Python）。

## 包

| 包 | 说明 | 状态 |
| --- | --- | --- |
| [karen-ai](karen-ai/) | 统一 LLM API 层：消息类型、事件流、Provider/Models 注册表、鉴权、Anthropic 与 OpenAI 兼容适配器。参考 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi) 的 Python 重写。 | 核心已完成 |
| karen-agent | （计划中）基于 karen-ai 的 agent 层 | 未开始 |

## 布局

```
karen/
├── karen-ai/          # 最底层：统一 LLM API
│   ├── src/karen_ai/  # 包源码
│   └── tests/         # pytest 测试
└── ...
```

每个包独立打包（`pyproject.toml`），可单独安装，也可 editable 安装：

```powershell
pip install -e ./karen-ai
pip install -e "./karen-ai[dev]"
cd karen-ai; pytest
```
