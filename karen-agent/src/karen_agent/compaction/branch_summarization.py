"""Branch summarization (pi's `harness/compaction/branch-summarization.ts`).

Generates the summary stored when a session navigates away from a branch.
Same deviations as `compaction.py`: no chord Context (explicit ``signal``
keyword), no assistant-call retry layer, details are plain camelCase dicts.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from karen_ai import AbortSignal, AssistantMessage, Context, Model, SimpleStreamOptions, TextContent, Usage, UserMessage
from karen_ai.types import KarenBase
from karen_ai.utils import content_text
from pydantic import ConfigDict

from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    message_field,
)
from ..result import BranchSummaryError, Err, Result, err, ok
from ..session.types import Branch, BranchScan, Entry, Session
from ..types import AgentMessage
from .compaction import (
    SUMMARIZATION_SYSTEM_PROMPT,
    SummaryRequest,
    create_summary_request_options,
    estimate_tokens,
)
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "BranchSummaryResult",
    "BranchPreparation",
    "CollectEntriesResult",
    "GenerateBranchSummaryOptions",
    "PreparedBranchSummaryOptions",
    "BRANCH_SUMMARY_PREAMBLE",
    "BRANCH_SUMMARY_PROMPT",
    "DEFAULT_BRANCH_RESERVE_TOKENS",
    "collect_entries_for_branch_summary",
    "prepare_branch_entries",
    "generate_branch_summary",
    "generate_branch_summary_with_request",
]

#: File-operation details stored on generated branch summary entries, as a
#: plain camelCase dict: ``{"readFiles": [...], "modifiedFiles": [...]}``.


class BranchSummaryResult(KarenBase):
    """Generated branch summary data ready to be persisted as a branch-summary entry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    usage: Optional[Usage] = None
    read_files: List[str]
    modified_files: List[str]


class BranchPreparation(KarenBase):
    """Prepared branch content for summarization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Messages selected for the branch summary.
    messages: List[AgentMessage]
    #: File operations extracted from the branch.
    file_ops: FileOperations
    #: Estimated token count for selected messages.
    total_tokens: int


class CollectEntriesResult(KarenBase):
    """Entries selected for branch summarization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Entries to summarize in chronological order.
    entries: List[Entry]
    #: Deepest common ancestor between the previous tip and target entry.
    common_ancestor_id: Optional[str] = None


class GenerateBranchSummaryOptions(KarenBase):
    """Options for generating a branch summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Provider collection the summarization request goes through; owns auth resolution.
    models: Any
    #: Model used for summarization.
    model: Model
    #: Optional instructions appended to or replacing the default prompt.
    custom_instructions: Optional[str] = None
    #: Replace the default prompt with custom instructions instead of appending them.
    replace_instructions: Optional[bool] = None
    #: Tokens reserved for prompt and model output. Defaults to 16384.
    reserve_tokens: Optional[int] = None


DEFAULT_BRANCH_RESERVE_TOKENS = 16384


async def collect_entries_for_branch_summary(
    branch: Branch,
    session: Session,
    old_tip_id: Optional[str],
    target_id: str,
) -> CollectEntriesResult:
    """Collect entries that should be summarized before navigating to a different session tree entry."""
    if not old_tip_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)
    old_path = {entry.id for entry in await branch.find_entries(BranchScan(start=old_tip_id))}
    target_path = await branch.find_entries(BranchScan(start=target_id))
    common_ancestor_id: Optional[str] = None
    for entry in target_path:
        if entry.id in old_path:
            common_ancestor_id = entry.id
            break
    entries: List[Entry] = []
    current: Optional[str] = old_tip_id

    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if entry is None:
            raise Exception(f"Corrupt session: entry {current} not found")
        entries.append(entry)
        current = entry.parent_id
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry(entry: Entry) -> Optional[AgentMessage]:
    if entry.type == "message":
        if message_field(entry.message, "role") == "toolResult":  # type: ignore[attr-defined]
            return None
        return entry.message  # type: ignore[attr-defined]
    if entry.type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)  # type: ignore[attr-defined]
    if entry.type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)  # type: ignore[attr-defined]
    return None


def prepare_branch_entries(entries: List[Entry], token_budget: int = 0) -> BranchPreparation:
    """Prepare branch entries for summarization within an optional token budget."""
    messages: List[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0

    for entry in entries:
        if entry.type != "branch_summary":
            continue
        details = getattr(entry, "details", None)
        if not isinstance(details, dict):
            continue
        read_files = details.get("readFiles")
        if isinstance(read_files, list):
            for path in read_files:
                if isinstance(path, str):
                    file_ops.read.add(path)
        modified_files = details.get("modifiedFiles")
        if isinstance(modified_files, list):
            for path in modified_files:
                if isinstance(path, str):
                    file_ops.edited.add(path)

    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in ("compaction", "branch_summary"):
                if total_tokens < token_budget * 0.9:
                    messages.insert(0, message)
                    total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: List[Entry],
    options: GenerateBranchSummaryOptions,
    *,
    signal: Optional[AbortSignal] = None,
) -> Result[BranchSummaryResult, BranchSummaryError]:
    """Generate a summary for abandoned branch entries."""
    reserve_tokens = options.reserve_tokens if options.reserve_tokens is not None else DEFAULT_BRANCH_RESERVE_TOKENS
    context_window = options.model.context_window or 128000
    preparation = prepare_branch_entries(entries, context_window - reserve_tokens)

    async def request(ai_context: Context, request_options: SimpleStreamOptions) -> AssistantMessage:
        return await options.models.complete_simple(
            options.model,
            ai_context,
            create_summary_request_options(request_options, signal=signal),
        )

    return await generate_branch_summary_with_request(
        preparation,
        PreparedBranchSummaryOptions(
            custom_instructions=options.custom_instructions,
            replace_instructions=options.replace_instructions,
        ),
        request,
        signal=signal,
    )


class PreparedBranchSummaryOptions(KarenBase):
    custom_instructions: Optional[str] = None
    replace_instructions: Optional[bool] = None


async def generate_branch_summary_with_request(
    preparation: BranchPreparation,
    options: PreparedBranchSummaryOptions,
    request: SummaryRequest,
    *,
    signal: Optional[AbortSignal] = None,
) -> Result[BranchSummaryResult, BranchSummaryError]:
    """Generate a prepared branch summary through a caller-owned one-request boundary."""
    messages = preparation.messages
    file_ops = preparation.file_ops
    if not messages:
        return ok(BranchSummaryResult(summary="No content to summarize", read_files=[], modified_files=[]))

    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    if options.replace_instructions and options.custom_instructions:
        instructions = options.custom_instructions
    elif options.custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {options.custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"

    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000)),
    ]
    response = await request(
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(SimpleStreamOptions(max_tokens=2048), signal=signal),
    )
    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                "summarization_failed", f"Branch summary failed: {response.error_message or 'Unknown error'}"
            )
        )

    summary = BRANCH_SUMMARY_PREAMBLE + content_text(response.content)
    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        BranchSummaryResult(
            summary=summary or "No summary generated",
            usage=response.usage,
            read_files=read_files,
            modified_files=modified_files,
        )
    )
