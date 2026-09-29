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
| `karen_ai.api` | `api/` | API 适配器：`anthropic_messages`、`openai_completions`、`openai_responses`、`openai_codex_responses`（SSE + WebSocket 双传输）、`azure_openai_responses`、`bedrock_converse_stream`（手写 SigV4 + eventstream）、`google_generative_ai`、`google_vertex`、`mistral_conversations`、`pi_messages`（Radius 网关协议） |
| `karen_ai.model_catalog` | `model-catalog.ts` + `providers/data/` |  vendored 静态模型目录（42 个 provider、约 1500 个模型，来自 pi-ai 生成产物），按类型 flatten |
| `karen_ai.providers` | `providers/` | 42 个内置 provider（与 pi-ai `builtinProviders()` 对齐）、通用 OpenAI 兼容工厂、faux 测试 provider |
| `karen_ai.telemetry` | `@earendil-works/pi-telemetry` | `TelemetryContext` / `TelemetrySpan` 协议、`NOOP_TELEMETRY_CONTEXT`、`InMemoryTelemetryContext`、`define_telemetry_schema` / `create_typed_span_starter` |
| `karen_ai.compat` + `karen_ai.api.legacy_aliases` | `compat.ts` / `legacy-api-aliases.ts` | 迁移期的全局 API：api 注册表、带环境变量注入的 `stream`/`complete`、静态目录读取、旧的逐 API 别名 |
| `karen_ai.session_resources` | `session-resources.ts` | 会话级资源清理钩子（`cleanup_session_resources`），Codex WebSocket 连接池挂在这里 |

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

对真实 API 的最小验证脚本见 `examples/deepseek_smoke.py`（读 `DEEPSEEK_API_KEY`，
`python examples/deepseek_smoke.py [--model deepseek-flash] ["问题"]`）。

## Provider

```python
from karen_ai.providers import (
    anthropic_provider,      # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_OAUTH_TOKEN / Claude Pro·Max OAuth
    openai_provider,         # OPENAI_API_KEY
    openai_codex_provider,   # ChatGPT Plus/Pro OAuth（订阅）
    deepseek_provider,       # DEEPSEEK_API_KEY
    openrouter_provider,     # OPENROUTER_API_KEY / OAuth，含图像生成与分类器
    github_copilot_provider, # COPILOT_GITHUB_TOKEN / GitHub OAuth（订阅）
    openai_compatible_provider,  # 任意 OpenAI 兼容端点（vLLM / llama.cpp / 代理）
    builtin_providers,       # 全部 42 个内置 provider
)

models = create_models()
for p in (anthropic_provider(), openai_provider(), deepseek_provider()):
    models.set_provider(p)

available = await models.get_available()   # 鉴权已配置的全部模型
```

内置 provider 的模型目录由 `scripts/generate_models.py` 生成到 `providers/data/*.json`
（pi-ai `scripts/generate-models.ts` 的 Python 移植：models.dev + OpenRouter + Vercel AI Gateway +
Nvidia NIM + Radius，再叠加 pi-ai 的各 provider 修正表）：

```powershell
python scripts/generate_models.py            # 重新生成全部目录（含 .manifest.json）
python scripts/generate_models.py --strict   # 上游失败 / 白名单不匹配直接报错
python scripts/generate_models.py --json-only --json-output out/   # 只出 JSON 目录
python scripts/check_model_data.py           # 校验已提交的目录与清单是否一致
```

动态目录 provider（如 Radius 网关）在运行时通过 `models.refresh()` 抓取并持久化。

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

**已移植**：类型系统、事件流、lazy_stream、transcript 回放、鉴权（api-key + OAuth 登录/刷新）、凭证/目录存储、Models 注册表、create_provider、全部 API 适配器（anthropic-messages / openai-completions / openai-responses / openai-codex-responses / azure-openai-responses / bedrock-converse-stream / google-generative-ai / google-vertex / mistral-conversations / pi-messages）、42 个内置 provider（生成目录 + OAuth 流程）、通用兼容工厂、图像生成（openrouter-images）、classifier（typesafe / cloudflare-workers-ai system-one）、Codex WebSocket 传输（会话复用 + `previous_response_id` 续传 + SSE 回退）、`compat.ts` / `legacy-api-aliases.ts` 旧全局 API、`generate-models.ts` 生成管线、pi-telemetry 运行时、faux 测试 provider。

**尚未移植**：无已知项。

与 pi-ai 的**有意差异**：
- `stream` 系列入口必须在运行中的事件循环里调用（`pi-ai` 同步返回流的设计在 Python 里通过后台 task 实现）。
- `packages/coding-agent`（karen-agent 将基于本层重建，而非逐行移植）。
- `file://` 与 base64 之外的图片编码细节、`scripts/generate-test-image.ts` 这类开发用脚本。

## 开发

```powershell
pip install -e ".[dev]"
pytest
```
