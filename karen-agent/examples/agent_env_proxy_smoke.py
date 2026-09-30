"""M6 verification: LocalExecutionEnv (real bash) + stream_proxy end-to-end.

Keyless by design — M6 is infrastructure:
- `LocalExecutionEnv` filesystem + shell ops against the real local bash
  (capture limits, spill, text line reader)
- `stream_proxy` driven by the `Agent` class against a scripted loopback proxy
  server speaking the documented wire protocol — proving the full
  Agent → proxy → events → state.messages path without any provider key

Run:  .venv/Scripts/python.exe karen-agent/examples/agent_env_proxy_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from karen_ai import Model, ModelCost, Usage, UserMessage
from karen_ai.types import TranscriptContext
from karen_agent import (
    Agent,
    AgentInitialState,
    LocalExecutionEnv,
    ProxyStreamOptions,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    execute_shell_with_capture,
    stream_proxy,
)
from karen_agent.result import Ok

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, condition))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


PROXY_MODEL = Model(
    id="proxy-model",
    name="proxy-model",
    api="proxy",
    provider="proxy",
    base_url="",
    reasoning=False,
    input=[],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=128000,
    max_tokens=4096,
)

ANSWER = "Aloha from the proxy"


class ScriptedProxyServer:
    """Speaks the proxy wire protocol: POST /api/stream → `data: ` JSON lines."""

    def __init__(self) -> None:
        self.requests = []
        self.server = None

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _handle(self, reader, writer):
        try:
            request_line = (await reader.readline()).decode().strip()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                name, _, value = line.decode().partition(":")
                headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length) if length else b""
            self.requests.append((request_line, headers, json.loads(body)))

            events = [
                {"type": "start"},
                {"type": "text_start", "contentIndex": 0},
                {"type": "text_delta", "contentIndex": 0, "delta": ANSWER[: len(ANSWER) // 2]},
                {"type": "text_delta", "contentIndex": 0, "delta": ANSWER[len(ANSWER) // 2 :]},
                {"type": "text_end", "contentIndex": 0},
                {
                    "type": "done",
                    "reason": "stop",
                    "usage": {
                        "input": 10,
                        "output": 5,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 15,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                    },
                },
            ]
            payload = "".join(f"data: {json.dumps(event)}\n" for event in events).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass


async def check_env(workdir: Path) -> None:
    env = LocalExecutionEnv(str(workdir))

    write = await env.write_file("nested/proof.txt", "env-proof-42\nsecond line\n")
    check("env.write_file", isinstance(write, Ok))
    read = await env.read_text_file("nested/proof.txt")
    check("env.read_text_file", isinstance(read, Ok) and read.value.startswith("env-proof-42"))

    lines = await env.read_text_lines("nested/proof.txt")
    check(
        "env.read_text_lines",
        isinstance(lines, Ok) and lines.value == ["env-proof-42", "second line"],
        repr(lines.value if isinstance(lines, Ok) else lines.error),
    )

    exec_result = await env.exec(
        "cat nested/proof.txt && echo $KAREN_SMOKE",
        ShellExecOptions(env={"KAREN_SMOKE": "merged-env"}),
    )
    output_ok = isinstance(exec_result, Ok) and exec_result.value.exit_code == 0
    check("env.exec exit code", output_ok)

    captured = await execute_shell_with_capture(env, "for i in $(seq 1 50); do echo line-$i; done")
    check(
        "execute_shell_with_capture",
        isinstance(captured, Ok) and "line-50" in captured.value.output and captured.value.exit_code == 0,
    )

    limited = await env.exec(
        "for i in $(seq 1 200); do echo row-$i; done",
        ShellExecOptions(
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(max_bytes=1024 * 1024, max_lines=20, retain="tail"),
                spill=True,
            )
        ),
    )
    truncated = isinstance(limited, Ok) and limited.value.truncation.truncated
    spilled = False
    if isinstance(limited, Ok) and limited.value.spill_path:
        spilled = "row-1\n" in Path(limited.value.spill_path).read_text(encoding="utf-8")
    check("env.exec truncation + spill", truncated and spilled, f"spill={limited.value.spill_path if isinstance(limited, Ok) else None}")

    await env.cleanup()


async def check_proxy_agent() -> None:
    async with ScriptedProxyServer() as server:
        def proxy_stream_fn(model, context, options):
            return stream_proxy(
                model,
                context,
                ProxyStreamOptions(auth_token="smoke-token", proxy_url=server.url, temperature=0.5),
            )

        agent = Agent(
            initial_state=AgentInitialState(
                system_prompt="You are a smoke-test agent. Answer briefly.",
                model=PROXY_MODEL,
                thinking_level="off",
                tools=[],
            ),
            stream_fn=proxy_stream_fn,
        )
        await agent.prompt("Say the greeting.")
        await agent.wait_for_idle()

        assistant_texts = [
            block.text
            for message in agent.state.messages
            if message.role == "assistant"
            for block in message.content
            if block.type == "text"
        ]
        check(
            "agent over stream_proxy",
            any(ANSWER in text for text in assistant_texts),
            repr(assistant_texts[-1]) if assistant_texts else "no assistant text",
        )
        check(
            "proxy received authorized request",
            len(server.requests) == 1
            and server.requests[0][0] == "POST /api/stream HTTP/1.1"
            and server.requests[0][1].get("authorization") == "Bearer smoke-token"
            and server.requests[0][2]["options"] == {"temperature": 0.5},
        )
        last = agent.state.messages[-1]
        usage_ok = last.role == "assistant" and last.usage.input == 10 and last.usage.total_tokens == 15
        check("proxy usage flowed through", usage_ok)
        check("agent error-free", agent.state.error_message is None)


async def main() -> int:
    print("== M6 env + proxy smoke (keyless) ==")
    with tempfile.TemporaryDirectory(prefix="karen-env-smoke-") as tmp:
        await check_env(Path(tmp))
    await check_proxy_agent()

    failed = [name for name, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
