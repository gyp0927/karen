"""Session persistence smoke test: create, append, fork, resume — no API key needed.

Run:  ../.venv/Scripts/python.exe examples/session_smoke.py
"""

import asyncio
import tempfile
import time

from karen_ai import UserMessage

from karen_agent.session import (
    BranchForkOptions,
    BranchScan,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    LaneConfiguration,
    LaneModelRef,
    LaneState,
    lane_config,
    lane_state,
)


def user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


async def main() -> None:
    root = tempfile.mkdtemp(prefix="karen-sessions-")
    repo = JsonlSessionRepo(root)
    cwd = "E:/karen"

    # Create a session with a configured "main" lane and append two messages.
    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd))
    branch = await session.create_branch("main", None)
    await session.set_value(
        lane_config("main"),
        LaneConfiguration(
            model=LaneModelRef(provider="deepseek", model_id="deepseek-v4-pro"),
            thinking_level="medium",
            active_tool_names=["read", "write"],
        ).model_dump(mode="json", by_alias=True),
    )
    await session.set_value(lane_state("main"), LaneState().model_dump(mode="json", by_alias=True))
    await session.set_name("smoke")
    first = await branch.append_message(user("hello"))
    second = await branch.append_message(user("how do sessions work?"))
    stats = await session.get_stats()
    print(f"created {session.metadata.id}: {stats.message_count} messages")
    print(f"  file: {session.metadata.path}")

    # Fork at the first message, then continue the fork independently.
    fork = await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=first))
    fork_branch = await fork.branch("main")
    await fork_branch.append_message(user("forked follow-up"))
    fork_entries = await fork_branch.find_entries(BranchScan(order="oldestFirst"))
    print(f"forked   {fork.metadata.id}: {[e.message.content for e in fork_entries]}")
    await fork.close()
    await session.close()

    # Resume by discovery: list sessions for the cwd, reopen, append.
    listed = await repo.list(JsonlSessionListOptions(cwd=cwd))
    print(f"listed   {[m.id for m in listed]}")
    resumed = await repo.open(next(m for m in listed if m.id == session.metadata.id))
    resumed_branch = await resumed.branch("main")
    await resumed_branch.append_message(user("resumed after restart"))
    entries = await resumed_branch.find_entries(BranchScan(order="oldestFirst"))
    print(f"resumed  {resumed.metadata.id}: {[e.message.content for e in entries]}")
    print(f"name={await resumed.get_name()!r} stats={(await resumed.get_stats()).message_count} messages")
    await resumed.close()


if __name__ == "__main__":
    asyncio.run(main())
