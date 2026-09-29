"""Minimal end-to-end check against the real DeepSeek API.

Credentials, in order of precedence:
  1. `~/.karen/credentials.json` — {"deepseek": {"type": "api_key", "key": "sk-..."}}
  2. the DEEPSEEK_API_KEY environment variable

Override the credential file with KAREN_CREDENTIALS_PATH.

Usage:
    python examples/deepseek_smoke.py
    python examples/deepseek_smoke.py --model deepseek-flash "用一句话解释什么是事件流"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from karen_ai import (
    Context,
    CreateModelsOptions,
    JsonFileCredentialStore,
    SimpleStreamOptions,
    UserMessage,
    calculate_cost,
    create_models,
)
from karen_ai.providers import deepseek_provider

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROMPT = "用一句话介绍你自己，并给出 2+3 的结果。"
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"


def _credentials_path() -> Path:
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    return Path(override).expanduser() if override else DEFAULT_CREDENTIALS


def _build_models():
    path = _credentials_path()
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "DEEPSEEK_API_KEY"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high"), default=None)
    args = parser.parse_args()

    models, source = _build_models()
    models.set_provider(deepseek_provider())

    model = models.get_model("deepseek", args.model)
    if model is None:
        ids = [m.id for m in models.get_models("deepseek")]
        print(f"unknown model {args.model!r}; available: {ids}", file=sys.stderr)
        return 2

    print(f"-> deepseek/{model.id}  auth from {source}  ctx={model.context_window}  max_tokens={model.max_tokens}")
    context = Context(messages=[UserMessage(content=args.prompt, timestamp=0)])

    if args.thinking:
        stream = models.stream_simple(model, context, SimpleStreamOptions(thinking_level=args.thinking))
    else:
        stream = models.stream_simple(model, context)

    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "thinking_delta":
            print(f"\033[2m{event.delta}\033[0m", end="", flush=True)
        elif event.type == "toolcall_delta":
            print(f"[toolcall {event.delta}]", end="", flush=True)
        elif event.type == "error":
            print(f"\n[stream error] {event.error.error_message}", file=sys.stderr)
    message = await stream.result()

    print("\n" + "-" * 60)
    usage = message.usage
    print(f"stop_reason={message.stop_reason}  usage={usage.model_dump(exclude_none=True)}")
    print(f"cost(total) = ${calculate_cost(model, usage).total:.6f}")
    if message.stop_reason == "error":
        print(f"FAILED: {message.error_message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
