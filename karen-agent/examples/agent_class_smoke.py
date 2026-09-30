"""M5 real-API check: the `Agent` class over DeepSeek.

- `Agent` drives the loop with the built-in tools (write/read/...)
- a loaded `SKILL.md` is offered to the model via the system prompt, and the
  model pulls it in by reading the file
- a steered message is injected mid-run and answered in the same run
- `format_skill_invocation` renders the explicit-invocation prompt

Run:  .venv/Scripts/python.exe karen-agent/examples/agent_class_smoke.py
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
from karen_agent import Agent, AgentInitialState, format_skill_invocation, format_skills_for_system_prompt, load_skills
from karen_agent.stream_fn import models_stream_fn
from karen_agent.tools import create_builtin_tools

MODEL_ID = os.environ.get("KAREN_MODEL", "deepseek-v4-pro")
CREDENTIALS = Path.home() / ".karen" / "credentials.json"
PROOF_FILE = "agent-class-proof.txt"


def now_ms() -> int:
    return int(time.time() * 1000)


def build_models():
    if CREDENTIALS.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(CREDENTIALS)))
    return create_models()


def write_skill(root: Path) -> None:
    skill_dir = root / "greeting"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: greeting\n"
        "description: Use when the user asks for the project greeting.\n"
        "---\n"
        "The project greeting is: Aloha from karen.\n",
        encoding="utf-8",
    )


def print_message_text(message) -> None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        print(content)
    elif isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text:
                print(text)


async def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="karen-agent-class-"))
    skills_root = workdir / "skills"
    write_skill(skills_root)
    loaded = load_skills(skills_root)
    if loaded.diagnostics:
        print(f"unexpected skill diagnostics: {loaded.diagnostics}", file=sys.stderr)
        return 1
    skills = loaded.skills
    print(f"workdir: {workdir}")
    print(f"loaded skills: {[skill.name for skill in skills]}")
    print("--- explicit invocation prompt ---")
    print(format_skill_invocation(skills[0], "Use it now."))
    print("----------------------------------")

    models = build_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", MODEL_ID)

    system_prompt = (
        "You are karen, a test agent. The working directory is {cwd}; "
        "relative tool paths resolve against it. Use the write tool to create files. "
        "Keep answers concise."
    ).format(cwd=workdir)
    system_prompt += "\n\n" + format_skills_for_system_prompt(skills)

    agent = Agent(
        stream_fn=models_stream_fn(models),
        initial_state=AgentInitialState(
            system_prompt=system_prompt,
            model=model,
            tools=create_builtin_tools(str(workdir)),
        ),
    )

    steered: list[bool] = []

    def on_event(event, signal) -> None:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[tool ->] {event.tool_name} {event.args}")
        elif event.type == "turn_end" and not steered:
            # Demonstrate live steering: queue a follow-up question for the same run.
            steered.append(True)
            agent.steer(
                UserMessage(
                    content="Now tell me the project greeting from the available skill (read its file).",
                    timestamp=now_ms(),
                )
            )
            print("\n[steering] queued a greeting question for the same run")

    agent.subscribe(on_event)

    print("\n=== run 1: write the proof file, then answer the steered question ===")
    await agent.prompt(
        f"Create a file named {PROOF_FILE} in the working directory containing exactly: m5 works"
    )
    await agent.wait_for_idle()

    print("\n\n=== state ===")
    print(f"messages: {len(agent.state.messages)} | is_streaming: {agent.state.is_streaming}")
    print(f"roles: {[getattr(m, 'role', None) for m in agent.state.messages]}")
    print(f"system prompt length: {len(agent.state.system_prompt)}")
    print(f"steering queue empty: {not agent.has_queued_messages()}")

    proof_path = workdir / PROOF_FILE
    proof_text = proof_path.read_text(encoding="utf-8").strip() if proof_path.exists() else None
    print(f"proof file exists: {proof_path.exists()} | content: {proof_text!r}")

    answers = []
    for message in agent.state.messages:
        content = getattr(message, "content", None)
        if getattr(message, "role", None) == "assistant" and isinstance(content, list):
            answers.extend(getattr(block, "text", "") or "" for block in content)
    combined = "\n".join(answers)

    checks = {
        "proof file content": proof_text == "m5 works",
        "steered question answered": "Aloha from karen" in combined,
        "no run error": agent.state.error_message is None,
    }
    print("\n=== checks ===")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
