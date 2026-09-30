"""End-to-end check of the M2 built-in tools against real DeepSeek.

The model gets the four built-in tools (read/write/edit/bash) rooted at a temp
directory and is asked to write a file, edit it, and verify it with bash. The
script asserts the file on disk actually went through both mutations.

Credentials: same precedence as examples/agent_smoke.py (credentials file, then
DEEPSEEK_API_KEY).

Usage:
    python examples/agent_tools_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, UserMessage, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import AgentContext, AgentLoopConfig, agent_loop, models_stream_fn
from karen_agent.tools import create_builtin_tools

MODEL_ID = "deepseek-v4-pro"
DEFAULT_CREDENTIALS = Path.home() / ".karen/credentials.json"

PROMPT = (
    'Call the write tool with path="note.txt" and content="karen tools v1". '
    'Then call the edit tool with path="note.txt" and edits=[{"oldText": "v1", "newText": "v2"}]. '
    'Then call the bash tool with command="cat note.txt". '
    "Finally, reply with the exact final content of note.txt and nothing else."
)


def _build_models():
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    path = Path(override).expanduser() if override else DEFAULT_CREDENTIALS
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "DEEPSEEK_API_KEY"


def _convert_to_llm(messages):
    # Keep system messages: the loop declares the tool loadout via tools_added
    # on a system message, so dropping "system" hides the tools from the model.
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


async def main() -> int:
    models, source = _build_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", MODEL_ID)
    print(f"-> deepseek/{model.id}  auth from {source}")

    workdir = tempfile.mkdtemp(prefix="karen-tools-")
    print(f"-> tool cwd: {workdir}")

    context = AgentContext(messages=[], tools=create_builtin_tools(workdir))
    config = AgentLoopConfig(model=model, convert_to_llm=_convert_to_llm)
    prompt = UserMessage(content=PROMPT, timestamp=int(time.time() * 1000))
    stream = agent_loop([prompt], context, config, None, models_stream_fn(models))

    tool_calls = []
    async for event in stream:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            tool_calls.append(event.tool_name)
            args = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in event.args.items()}
            print(f"\n[tool ->] {event.tool_name}({args})")
        elif event.type == "tool_execution_end":
            text = "".join(getattr(c, "text", "") for c in event.result.content)
            print(f"[tool <-] {event.tool_name}: {text[:200]!r}{' (error)' if event.is_error else ''}")
    print()
    messages = await stream.result()

    note = Path(workdir) / "note.txt"
    final = messages[-1]
    answer = "".join(getattr(c, "text", "") for c in getattr(final, "content", []))
    print("-" * 60)
    print(f"tool calls: {tool_calls}")
    print(f"file on disk: {note.read_bytes()!r}" if note.exists() else "file on disk: MISSING")
    print(f"final answer: {answer!r}")

    if getattr(final, "stop_reason", None) == "error":
        print(f"FAILED: {final.error_message}", file=sys.stderr)
        return 1
    if not note.exists():
        print("FAILED: note.txt was never written", file=sys.stderr)
        return 1
    content = note.read_text(encoding="utf-8")
    if "karen tools v2" not in content:
        print(f"FAILED: expected edited content 'karen tools v2', got {content!r}", file=sys.stderr)
        return 1
    if "edit" not in tool_calls or "bash" not in tool_calls:
        print(f"FAILED: expected edit+bash tool calls, got {tool_calls}", file=sys.stderr)
        return 1
    print("OK: write → edit → bash all executed against the real model; file content is correct")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
