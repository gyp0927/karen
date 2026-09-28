# karen-ai

统一 LLM API 层 — [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi) 核心层的 Python 重写。

一条消息类型系统 + 一个事件流协议，对接不同的 provider API；上层（如 karen-agent）只面对统一接口。

## 架构

```
Context ──normalize_context()──▶ TranscriptContext ──▶ Provider ──▶ API 适配器 ──▶ LLM API
                                      │                                │
                                      ▼                                ▼
                          Models 注册表（鉴权、目录）          AssistantMessageEventStream
```

| 模块 | 对应 pi-ai | 内容 |
| --- | --- | --- |
| `karen_ai.types` | `types.ts` | 消息/内容块/Usage/模型目录/事件协议/选项，pydantic 模型，JSON 字段名与 pi-ai 的 camelCase 线格式一致 |
| `karen_ai.event_stream` | `utils/event-stream.ts` | `EventStream` / `AssistantMessageEventStream`（asyncio.Queue 实现，`async for` + `await result()`） |
| `karen_ai.lazy` | `api/lazy.ts` | `lazy_stream`：同步返回流，后台异步完成鉴权与分发，失败以 error 事件收尾 |
| `karen_ai.transcript` | `utils/transcript.ts` | 系统消息回放（sections、tools_added/removed）、`normalize_context`、跨 provider 折叠 |
| `karen_ai.auth` | `auth/` | 凭证（api_key/OAuth）、`CredentialStore`（内存 + JSON 文件）、鉴权解析（OAuth 到期自动加锁刷新） |
| `karen_ai.models` | `models.ts` | `Provider`、`Models` 注册表、`create_provider`、`calculate_cost`、thinking 级别映射 |
| `karen_ai.models_store` | `models-store.ts` | 模型目录持久化（内存 + JSON 文件） |
| `karen_ai.api` | `api/` | API 适配器：`anthropic_messages`、`openai_completions`（含兼容自动探测与各家 thinking 格式） |
| `karen_ai.providers` | `providers/` | 内置 provider：anthropic / openai / deepseek / openrouter / 通用 OpenAI 兼容工厂 / faux 测试 provider |

## 快速开始

```python
import asyncio
from karen_ai import Context, UserMessage, create_models
from karen_ai.providers import anthropic_provider

async def main():
    models = create_models()
    models.set_provider(anthropic_provider())   # 读 ANTHROPIC_API_KEY

    model = models.get_model("anthropic", "claude-sonnet-5")
    context = Context(messages=[UserMessage(content="你好", timestamp=0)])

    # 流式
    stream = models.stream_simple(model, context)
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    message = await stream.result()

    # 或一次性
    message = await models.complete_simple(model, context)

asyncio.run(main())
```

注意：所有 stream 入口都必须在运行中的事件循环里调用（asyncio 生态约束，`pi-ai` 中同步返回流的设计在 Python 里通过后台 task 实现）。

## Provider

```python
from karen_ai.providers import (
    anthropic_provider,      # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_OAUTH_TOKEN
    openai_provider,         # OPENAI_API_KEY
    deepseek_provider,       # DEEPSEEK_API_KEY
    openrouter_provider,     # OPENROUTER_API_KEY
    openai_compatible_provider,  # 任意 OpenAI 兼容端点（vLLM / llama.cpp / 代理）
)

models = create_models()
for p in (anthropic_provider(), openai_provider(), deepseek_provider()):
    models.set_provider(p)

available = await models.get_available()   # 鉴权已配置的全部模型
```

自定义 OpenAI 兼容端点：

```python
from karen_ai.providers import openai_compatible_provider
from karen_ai import Model, ModelCost

vllm = openai_compatible_provider(
    id="local-vllm",
    base_url="http://localhost:8000/v1",
    models=[Model(id="qwen3-32b", name="Qwen3 32B", api="openai-completions",
                  provider="local-vllm", base_url="http://localhost:8000/v1",
                  cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                  reasoning=True, context_window=32768, max_tokens=8192)],
)
```

## 与 pi-ai 的范围差异

**已移植（核心层）**：类型系统、事件流、lazy_stream、transcript 回放、鉴权（api-key + OAuth 刷新逻辑）、凭证/目录存储、Models 注册表、create_provider、`anthropic-messages` 与 `openai-completions` 适配器（流式、工具调用、thinking、缓存控制、compat 自动探测、跨 provider 消息变换）、四个内置 provider、通用兼容工厂、faux 测试 provider。

**暂未移植（后续按需补）**：
- `openai-responses` / `azure` / `codex` / `bedrock` / `google` / `mistral` 等其余 API 适配器
- 各 provider 的 OAuth 登录流程（类型与刷新机制已就绪）
- 动态模型目录抓取脚本（pi-ai 的 `generate-models.ts` 产物）；内置 provider 使用手写精简目录
- deferred response、图像生成、classifier（类型已定义，调用会返回明确的 error 结果）
- `compat.ts` 旧全局 API、遥测

## 开发

```powershell
pip install -e ".[dev]"
pytest
```
