"""End-to-end agent loop check against the real DeepSeek API.

One round of "question -> tool call -> answer": the model is asked a question it
can only answer by calling the echo tool, the loop executes the tool, and the
model summarizes the tool result.

Credentials, in order of precedence (same as karen-ai's examples/deepseek_smoke.py):
  1. `~/.karen/credentials.json` — {"deepseek": {"type": "api_key", "key": "sk-..."}}
  2. the DEEPSEEK_API_KEY environment variable

Usage:
    python examples/agent_smoke.py
    python examples/agent_smoke.py --model deepseek-flash "用 echo 工具复述：karen 你好"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, Message, TextContent, UserMessage, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    agent_loop,
    models_stream_fn,
)

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROMPT = "Call the echo tool with value=\"karen-agent works\", then tell me what the tool returned."
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"


def _credentials_path() -> Path:
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    return Path(override).expanduser() if override else DEFAULT_CREDENTIALS


def _build_models():
    path = _credentials_path()
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "DEEPSEEK_API_KEY"


def _echo_tool() -> AgentTool:
    async def execute(tool_call_id, params, signal, on_update):
        return AgentToolResult(
            content=[TextContent(text=f"echo: {params['value']}")],
            details={"echoed": params["value"]},
        )

    return AgentTool(
        name="echo",
        description="Echo back the given value verbatim.",
        label="Echo",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "description": "The text to echo back"}},
            "required": ["value"],
        },
        execute=execute,
    )


def _convert_to_llm(messages):
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    models, source = _build_models()
    models.set_provider(deepseek_provider())

    model = models.get_model("deepseek", args.model)
    if model is None:
        ids = [m.id for m in models.get_models("deepseek")]
        print(f"unknown model {args.model!r}; available: {ids}", file=sys.stderr)
        return 2

    print(f"-> deepseek/{model.id}  auth from {source}")

    context = AgentContext(messages=[], tools=[_echo_tool()])
    config = AgentLoopConfig(model=model, convert_to_llm=_convert_to_llm)
    prompt = UserMessage(content=args.prompt, timestamp=0)

    stream = agent_loop([prompt], context, config, None, models_stream_fn(models))
    async for event in stream:
        if event.type == "message_update":
            update = event.assistant_message_event
            if update.type == "text_delta":
                print(update.delta, end="", flush=True)
            elif update.type == "thinking_delta":
                print(f"\033[2m{update.delta}\033[0m", end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[tool ->] {event.tool_name}({event.args})")
        elif event.type == "tool_execution_end":
            status = "error" if event.is_error else "ok"
            text = "".join(getattr(c, "text", "") for c in event.result.content)
            print(f"[tool <-] {event.tool_name} {status}: {text}")

    messages = await stream.result()
    print("\n" + "-" * 60)
    last = messages[-1]
    if getattr(last, "stop_reason", None) == "error":
        print(f"FAILED: {last.error_message}", file=sys.stderr)
        return 1

    tool_results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
    print(f"messages={len(messages)}  tool_results={len(tool_results)}  stop_reason={last.stop_reason}")
    if not tool_results:
        print("FAILED: the model never called the echo tool", file=sys.stderr)
        return 1
    if any(m.is_error for m in tool_results):
        print("FAILED: a tool result is an error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
