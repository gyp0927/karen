"""Conversation compaction (pi's `harness/compaction/compaction.ts`).

Deviations from pi, consistent with M1/M2:

- pi's chord ``Context`` parameter is dropped; an optional ``signal`` keyword
  threads the abort signal into summary requests instead.
- ``details`` payloads are plain camelCase JSON dicts, per the M2 convention.

Summary calls go through ``retry_assistant_call`` exactly like pi's
``completeSimpleWithRetries``: callers pass an optional ``retry`` policy and
``callbacks``, and every provider request of one compaction (history summary
and turn-prefix summary alike) is retried inside that call.
"""

from __future__ import annotations

import math
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from karen_ai import (
    AbortSignal,
    AssistantMessage,
    Context,
    Model,
    RetryCallbacks,
    RetryPolicy,
    SimpleStreamOptions,
    TextContent,
    Usage,
    UserMessage,
    retry_assistant_call,
)
from karen_ai.types import KarenBase, ThinkingLevel
from karen_ai.utils import content_text
from pydantic import ConfigDict, Field

from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    message_field,
    safe_json_stringify,
)
from ..result import CompactionError, Err, Ok, Result, err, ok
from ..session.context import build_context_entries, session_entry_to_context_messages
from ..session.ids import uuid7
from ..session.types import Entry, MessageEntry
from ..types import AgentMessage
from ..utils.usage import add_usage
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "CompactionSettings",
    "DEFAULT_COMPACTION_SETTINGS",
    "CompactionDetails",
    "CompactResult",
    "CompactGenerationOptions",
    "CompactionPreparation",
    "ContextUsageEstimate",
    "CutPointResult",
    "GeneratedSummary",
    "RetryCallbacks",
    "RetryPolicy",
    "SummaryGenerationOptions",
    "SummaryRequest",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "ESTIMATED_IMAGE_CHARS",
    "create_summary_request_options",
    "complete_summary",
    "calculate_context_tokens",
    "get_last_assistant_usage",
    "estimate_context_tokens",
    "should_compact",
    "estimate_tokens",
    "find_turn_start_index",
    "find_cut_point",
    "generate_summary",
    "generate_summary_with_usage",
    "generate_summary_with_request",
    "prepare_compaction",
    "compact",
    "compact_with_request",
    "serialize_conversation",
]

#: One caller-owned summarization request: (ai context, stream options) -> assistant message.
SummaryRequest = Callable[[Context, SimpleStreamOptions], Awaitable[AssistantMessage]]


def create_summary_request_options(
    options: SimpleStreamOptions, *, signal: Optional[AbortSignal] = None
) -> SimpleStreamOptions:
    """Summaries are standalone requests: pin the abort signal, disable cache
    writes that cannot be reused, and tag a fresh session id."""
    return options.model_copy(
        update={
            "signal": signal,
            "cache_retention": "none",
            "session_id": options.session_id or uuid7(),
        }
    )


async def complete_summary(
    models: Any,
    model: Model,
    ai_context: Context,
    options: SimpleStreamOptions,
    *,
    signal: Optional[AbortSignal] = None,
    retry: Optional[RetryPolicy] = None,
    callbacks: Optional[RetryCallbacks] = None,
) -> AssistantMessage:
    """Default request boundary used by ``compact`` / ``generate_summary_*``.

    pi's `completeSimpleWithRetries`: summaries are standalone requests, so the
    retry policy and its callbacks apply to this single provider call.
    """
    request_options = create_summary_request_options(options, signal=signal)
    return await retry_assistant_call(
        lambda: models.complete_simple(model, ai_context, request_options),
        retry,
        signal=request_options.signal,
        callbacks=callbacks,
    )


class CompactionSettings(KarenBase):
    """Compaction thresholds and retention settings."""

    #: Enable automatic compaction decisions.
    enabled: bool = True
    #: Tokens reserved for summary prompt and output.
    reserve_tokens: int = 16384
    #: Approximate recent-context tokens to keep after compaction.
    keep_recent_tokens: int = 20000


#: Default compaction settings used by the harness.
DEFAULT_COMPACTION_SETTINGS = CompactionSettings()

#: File-operation details stored on generated compaction entries
#: (plain camelCase dict: ``{"readFiles": [...], "modifiedFiles": [...]}``).
CompactionDetails = Dict[str, List[str]]


def calculate_context_tokens(usage: Usage) -> int:
    """Calculate total context tokens from provider usage."""
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _get_assistant_usage(message: AgentMessage) -> Optional[Usage]:
    if message_field(message, "role") != "assistant":
        return None
    if message_field(message, "stop_reason", "stopReason") in ("aborted", "error"):
        return None
    usage = message_field(message, "usage")
    if usage is not None and calculate_context_tokens(usage) > 0:
        return usage
    return None


def get_last_assistant_usage(entries: List[Entry]) -> Optional[Usage]:
    """Return usage from the last valid assistant message in session entries."""
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.type == "message":
            usage = _get_assistant_usage(entry.message)  # type: ignore[attr-defined]
            if usage is not None:
                return usage
    return None


class ContextUsageEstimate(KarenBase):
    """Estimated context-token usage for a message list."""

    #: Estimated total context tokens.
    tokens: int
    #: Tokens reported by the most recent assistant usage block.
    usage_tokens: int
    #: Estimated tokens after the most recent assistant usage block.
    trailing_tokens: int
    #: Index of the message that provided usage, or None when none exists.
    last_usage_index: Optional[int] = None


def _get_last_assistant_usage_info(messages: List[AgentMessage]) -> Optional[tuple]:
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage is not None:
            return usage, i
    return None


def estimate_context_tokens(messages: List[AgentMessage]) -> ContextUsageEstimate:
    """Estimate context tokens for messages using provider usage when available."""
    usage_info = _get_last_assistant_usage_info(messages)

    if usage_info is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

    usage, index = usage_info
    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = sum(estimate_tokens(messages[i]) for i in range(index + 1, len(messages)))

    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    """Return whether context usage exceeds the configured compaction threshold."""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    chars = 0
    for block in content:
        block_type = message_field(block, "type")
        if block_type == "text":
            text = message_field(block, "text")
            if text:
                chars += len(text)
        elif block_type == "image":
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """Estimate token count for one message using a conservative character heuristic."""
    role = message_field(message, "role")

    if role == "user":
        chars = _estimate_text_and_image_content_chars(message_field(message, "content", default=""))
        return math.ceil(chars / 4)
    if role == "assistant":
        chars = 0
        content = message_field(message, "content", default=[])
        for block in content if isinstance(content, list) else []:
            block_type = message_field(block, "type")
            if block_type == "text":
                chars += len(message_field(block, "text", default="") or "")
            elif block_type == "thinking":
                chars += len(message_field(block, "thinking", default="") or "")
            elif block_type == "toolCall":
                chars += len(message_field(block, "name", default="") or "")
                chars += len(safe_json_stringify(message_field(block, "arguments")))
        return math.ceil(chars / 4)
    if role in ("custom", "toolResult"):
        chars = _estimate_text_and_image_content_chars(message_field(message, "content", default=""))
        return math.ceil(chars / 4)
    if role == "bashExecution":
        chars = len(message_field(message, "command", default="") or "") + len(
            message_field(message, "output", default="") or ""
        )
        return math.ceil(chars / 4)
    if role in ("branchSummary", "compactionSummary"):
        return math.ceil(len(message_field(message, "summary", default="") or "") / 4)
    return 0


def _find_valid_cut_points(entries: List[Entry], start_index: int, end_index: int) -> List[int]:
    cut_points: List[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.type == "message":
            role = message_field(entry.message, "role")  # type: ignore[attr-defined]
            if role in ("bashExecution", "custom", "branchSummary", "compactionSummary", "user", "assistant"):
                cut_points.append(i)
            # toolResult messages are never valid cut points
        if entry.type == "branch_summary":
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: List[Entry], entry_index: int, start_index: int) -> int:
    """Find the user-visible message that starts the turn containing an entry."""
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.type == "branch_summary":
            return i
        if entry.type == "message":
            role = message_field(entry.message, "role")  # type: ignore[attr-defined]
            if role in ("user", "bashExecution"):
                return i
    return -1


class CutPointResult(KarenBase):
    """Cut point selected for compaction."""

    #: Index of the first entry retained after compaction.
    first_kept_entry_index: int
    #: Index of the turn-start entry when the cut splits a turn, otherwise -1.
    turn_start_index: int
    #: Whether the selected cut point splits an in-progress turn.
    is_split_turn: bool


def find_cut_point(
    entries: List[Entry], start_index: int, end_index: int, keep_recent_tokens: int
) -> CutPointResult:
    """Find the compaction cut point that keeps approximately the requested recent-token budget."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.type != "message":
            continue
        accumulated_tokens += estimate_tokens(entry.message)  # type: ignore[attr-defined]
        if accumulated_tokens >= keep_recent_tokens:
            for cut_point in cut_points:
                if cut_point >= i:
                    cut_index = cut_point
                    break
            break

    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.type in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == "message" and message_field(cut_entry.message, "role") == "user"  # type: ignore[attr-defined]
    turn_start_index = -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not is_user_message and turn_start_index != -1,
    )


SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


class GeneratedSummary(KarenBase):
    """Summary text plus the provider usage of the call that produced it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    usage: Usage


class SummaryGenerationOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    reserve_tokens: int
    custom_instructions: Optional[str] = None
    previous_summary: Optional[str] = None
    thinking_level: Optional[str] = None


def _completion_options(model: Model, max_tokens: int, thinking_level: Optional[str]) -> SimpleStreamOptions:
    if model.reasoning and thinking_level and thinking_level != "off":
        return SimpleStreamOptions(max_tokens=max_tokens, reasoning=thinking_level)
    return SimpleStreamOptions(max_tokens=max_tokens)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def generate_summary(
    current_messages: List[AgentMessage],
    models: Any,
    model: Model,
    reserve_tokens: int,
    custom_instructions: Optional[str] = None,
    previous_summary: Optional[str] = None,
    thinking_level: Optional[str] = None,
    *,
    signal: Optional[AbortSignal] = None,
    retry: Optional[RetryPolicy] = None,
    callbacks: Optional[RetryCallbacks] = None,
) -> Result[str, CompactionError]:
    """Generate or update a conversation summary for compaction."""
    result = await generate_summary_with_usage(
        current_messages,
        models,
        model,
        reserve_tokens,
        custom_instructions,
        previous_summary,
        thinking_level,
        signal=signal,
        retry=retry,
        callbacks=callbacks,
    )
    if isinstance(result, Err):
        return err(result.error)
    return ok(result.value.text)


async def generate_summary_with_usage(
    current_messages: List[AgentMessage],
    models: Any,
    model: Model,
    reserve_tokens: int,
    custom_instructions: Optional[str] = None,
    previous_summary: Optional[str] = None,
    thinking_level: Optional[str] = None,
    *,
    signal: Optional[AbortSignal] = None,
    retry: Optional[RetryPolicy] = None,
    callbacks: Optional[RetryCallbacks] = None,
) -> Result[GeneratedSummary, CompactionError]:
    """Generate or update a conversation summary and return its provider usage."""

    async def request(ai_context: Context, options: SimpleStreamOptions) -> AssistantMessage:
        return await complete_summary(
            models, model, ai_context, options, signal=signal, retry=retry, callbacks=callbacks
        )

    return await generate_summary_with_request(
        current_messages,
        SummaryGenerationOptions(
            model=model,
            reserve_tokens=reserve_tokens,
            custom_instructions=custom_instructions,
            previous_summary=previous_summary,
            thinking_level=thinking_level,
        ),
        request,
        signal=signal,
    )


async def generate_summary_with_request(
    current_messages: List[AgentMessage],
    options: SummaryGenerationOptions,
    request: SummaryRequest,
    *,
    signal: Optional[AbortSignal] = None,
) -> Result[GeneratedSummary, CompactionError]:
    """Generate one summary through a caller-owned one-request boundary."""
    model = options.model
    max_tokens = math.floor(0.8 * options.reserve_tokens)
    if model.max_tokens > 0:
        max_tokens = min(max_tokens, model.max_tokens)
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if options.previous_summary else SUMMARIZATION_PROMPT
    if options.custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {options.custom_instructions}"
    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if options.previous_summary:
        prompt_text += f"<previous-summary>\n{options.previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=_now_ms()),
    ]

    response = await request(
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(
            _completion_options(model, max_tokens, options.thinking_level), signal=signal
        ),
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed", f"Summarization failed: {response.error_message or 'Unknown error'}"
            )
        )

    return ok(GeneratedSummary(text=content_text(response.content), usage=response.usage))


class CompactionPreparation(KarenBase):
    """Prepared inputs for a compaction run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Messages summarized into the history summary.
    messages_to_summarize: List[AgentMessage]
    #: Prefix messages summarized separately when compaction splits a turn.
    turn_prefix_messages: List[AgentMessage]
    #: Recent messages retained after compaction and stored on the compaction entry.
    retained_tail: List[AgentMessage]
    #: Whether compaction splits a turn.
    is_split_turn: bool
    #: Estimated context tokens before compaction.
    tokens_before: int
    #: Previous compaction summary used for iterative updates.
    previous_summary: Optional[str] = None
    #: File operations extracted from summarized history.
    file_ops: FileOperations
    #: Settings used to prepare compaction.
    settings: CompactionSettings


def _get_message_from_entry(entry: Entry) -> Optional[AgentMessage]:
    if entry.type == "message":
        return entry.message  # type: ignore[attr-defined]
    if entry.type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)  # type: ignore[attr-defined]
    if entry.type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)  # type: ignore[attr-defined]
    return None


def _get_message_from_entry_for_compaction(entry: Entry) -> Optional[AgentMessage]:
    if entry.type == "compaction":
        return None
    return _get_message_from_entry(entry)


def _extract_file_operations(
    messages: List[AgentMessage], entries: List[Entry], prev_compaction_index: int
) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        details = getattr(prev_compaction, "details", None)
        if isinstance(details, dict):
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
    for message in messages:
        extract_file_ops_from_message(message, file_ops)
    return file_ops


def prepare_compaction(
    path_entries: List[Entry], settings: CompactionSettings
) -> Result[Optional[CompactionPreparation], CompactionError]:
    """Prepare session entries for compaction, or Ok(None) when compaction is not applicable."""
    if not path_entries or path_entries[-1].type == "compaction":
        return ok(None)

    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].type == "compaction":
            prev_compaction_index = i
            break

    previous_summary: Optional[str] = None
    compactable_entries: List[Entry] = list(path_entries)
    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        previous_summary = prev_compaction.summary  # type: ignore[attr-defined]
        virtual_retained_entries = [
            MessageEntry(
                id=f"{prev_compaction.id}:retained:{index}",  # type: ignore[attr-defined]
                parent_id=prev_compaction.id if index == 0 else f"{prev_compaction.id}:retained:{index - 1}",  # type: ignore[attr-defined]
                seq=prev_compaction.seq,  # type: ignore[attr-defined]
                timestamp=message_field(message, "timestamp", default=0),
                message=message,
            )
            for index, message in enumerate(prev_compaction.retained_tail)  # type: ignore[attr-defined]
        ]
        compactable_entries = [*virtual_retained_entries, *path_entries[prev_compaction_index + 1 :]]

    boundary_end = len(compactable_entries)

    tokens_before = estimate_context_tokens(
        [m for entry in build_context_entries(path_entries) for m in session_entry_to_context_messages(entry)]
    ).tokens

    cut_point = find_cut_point(compactable_entries, 0, boundary_end, settings.keep_recent_tokens)
    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    messages_to_summarize: List[AgentMessage] = []
    for i in range(history_end):
        message = _get_message_from_entry_for_compaction(compactable_entries[i])
        if message is not None:
            messages_to_summarize.append(message)

    turn_prefix_messages: List[AgentMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            message = _get_message_from_entry_for_compaction(compactable_entries[i])
            if message is not None:
                turn_prefix_messages.append(message)

    retained_tail: List[AgentMessage] = []
    for i in range(cut_point.first_kept_entry_index, boundary_end):
        message = _get_message_from_entry_for_compaction(compactable_entries[i])
        if message is not None:
            retained_tail.append(message)

    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for message in turn_prefix_messages:
            extract_file_ops_from_message(message, file_ops)

    return ok(
        CompactionPreparation(
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            retained_tail=retained_tail,
            is_split_turn=cut_point.is_split_turn,
            tokens_before=tokens_before,
            previous_summary=previous_summary,
            file_ops=file_ops,
            settings=settings,
        )
    )


class CompactResult(KarenBase):
    """Generated compaction data ready to be persisted as a compaction entry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Summary text that replaces compacted history in future context.
    summary: str
    #: Estimated context tokens before compaction.
    tokens_before: int
    #: Usage from the LLM call(s) that generated this summary, if available.
    usage: Optional[Usage] = None
    #: Retained recent messages stored directly on the compaction entry.
    retained_tail: List[AgentMessage] = Field(default_factory=list)
    #: Implementation-specific details stored with the compaction entry
    #: (plain camelCase dict, see :data:`CompactionDetails`).
    details: Optional[CompactionDetails] = None


class CompactGenerationOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    custom_instructions: Optional[str] = None
    thinking_level: Optional[str] = None


async def compact(
    preparation: CompactionPreparation,
    models: Any,
    model: Model,
    custom_instructions: Optional[str] = None,
    thinking_level: Optional[str] = None,
    *,
    signal: Optional[AbortSignal] = None,
    retry: Optional[RetryPolicy] = None,
    callbacks: Optional[RetryCallbacks] = None,
) -> Result[CompactResult, CompactionError]:
    """Generate compaction summary data from prepared session history."""

    async def request(ai_context: Context, options: SimpleStreamOptions) -> AssistantMessage:
        return await complete_summary(
            models, model, ai_context, options, signal=signal, retry=retry, callbacks=callbacks
        )

    return await compact_with_request(
        preparation,
        CompactGenerationOptions(
            model=model, custom_instructions=custom_instructions, thinking_level=thinking_level
        ),
        request,
        signal=signal,
    )


async def compact_with_request(
    preparation: CompactionPreparation,
    options: CompactGenerationOptions,
    request: SummaryRequest,
    *,
    signal: Optional[AbortSignal] = None,
) -> Result[CompactResult, CompactionError]:
    """Generate compaction data through a caller-owned boundary for each provider request."""
    summary: str
    summary_usage: Usage

    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Optional[Usage] = None
        if preparation.messages_to_summarize:
            history_result = await generate_summary_with_request(
                preparation.messages_to_summarize,
                SummaryGenerationOptions(
                    model=options.model,
                    reserve_tokens=preparation.settings.reserve_tokens,
                    custom_instructions=options.custom_instructions,
                    previous_summary=preparation.previous_summary,
                    thinking_level=options.thinking_level,
                ),
                request,
                signal=signal,
            )
            if isinstance(history_result, Err):
                return err(history_result.error)
            history_text = history_result.value.text
            history_usage = history_result.value.usage
        turn_prefix_result = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            options.model,
            preparation.settings.reserve_tokens,
            options.thinking_level,
            request,
            signal=signal,
        )
        if isinstance(turn_prefix_result, Err):
            return err(turn_prefix_result.error)
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_result.value.text}"
        summary_usage = (
            add_usage(history_usage, turn_prefix_result.value.usage)
            if history_usage is not None
            else turn_prefix_result.value.usage
        )
    else:
        summary_result = await generate_summary_with_request(
            preparation.messages_to_summarize,
            SummaryGenerationOptions(
                model=options.model,
                reserve_tokens=preparation.settings.reserve_tokens,
                custom_instructions=options.custom_instructions,
                previous_summary=preparation.previous_summary,
                thinking_level=options.thinking_level,
            ),
            request,
            signal=signal,
        )
        if isinstance(summary_result, Err):
            return err(summary_result.error)
        summary = summary_result.value.text
        summary_usage = summary_result.value.usage

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    details: CompactionDetails = {"readFiles": read_files, "modifiedFiles": modified_files}

    return ok(
        CompactResult(
            summary=summary,
            tokens_before=preparation.tokens_before,
            usage=summary_usage,
            retained_tail=preparation.retained_tail,
            details=details,
        )
    )


async def _generate_turn_prefix_summary(
    messages: List[AgentMessage],
    model: Model,
    reserve_tokens: int,
    thinking_level: Optional[str],
    request: SummaryRequest,
    *,
    signal: Optional[AbortSignal] = None,
) -> Result[GeneratedSummary, CompactionError]:
    max_tokens = math.floor(0.5 * reserve_tokens)
    if model.max_tokens > 0:
        max_tokens = min(max_tokens, model.max_tokens)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=_now_ms()),
    ]

    response = await request(
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(_completion_options(model, max_tokens, thinking_level), signal=signal),
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )

    return ok(GeneratedSummary(text=content_text(response.content), usage=response.usage))
