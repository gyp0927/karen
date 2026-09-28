"""Cross-provider transcript transform, mirroring pi-ai's api/transform-messages.ts.

- Downgrades images to text placeholders for non-vision models
- Converts thinking blocks to plain text for cross-model replay (keeping
  signed thinking for the same model)
- Normalizes tool call ids (and rewrites matching tool results)
- Inserts synthetic error tool results for orphaned tool calls
- Drops errored/aborted assistant messages
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ..types import (
    AssistantMessage,
    ContentBlock,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


def _replace_images_with_placeholder(content: List[ContentBlock], placeholder: str) -> List[TextContent]:
    result: List[TextContent] = []
    previous_was_placeholder = False

    for block in content:
        if isinstance(block, ImageContent):
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue
        if isinstance(block, TextContent):
            result.append(block)
            previous_was_placeholder = block.text == placeholder

    return result


def _downgrade_unsupported_images(messages: List[Message], model: Model) -> List[Message]:
    if "image" in model.input:
        return messages

    downgraded: List[Message] = []
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, list):
            downgraded.append(
                msg.model_copy(update={"content": _replace_images_with_placeholder(msg.content, NON_VISION_USER_IMAGE_PLACEHOLDER)})
            )
        elif msg.role == "toolResult":
            downgraded.append(
                msg.model_copy(update={"content": _replace_images_with_placeholder(msg.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER)})
            )
        else:
            downgraded.append(msg)
    return downgraded


def transform_messages(
    messages: List[Message],
    model: Model,
    normalize_tool_call_id: Optional[Callable[[str], str]] = None,
) -> List[Message]:
    """Normalize a transcript for replay against `model`."""
    tool_call_id_map: Dict[str, str] = {}
    image_aware = _downgrade_unsupported_images(list(messages), model)

    transformed: List[Message] = []
    for msg in image_aware:
        if msg.role in ("system", "user"):
            transformed.append(msg)
            continue

        if msg.role == "toolResult":
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                transformed.append(msg.model_copy(update={"tool_call_id": normalized_id}))
            else:
                transformed.append(msg)
            continue

        if msg.role == "assistant":
            assistant_msg: AssistantMessage = msg  # type: ignore[assignment]
            is_same_model = (
                assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
                and assistant_msg.model == model.id
            )

            transformed_content: List[ContentBlock] = []
            for block in assistant_msg.content:
                if block.type == "thinking":
                    # Redacted thinking is opaque encrypted content, only valid for the same model.
                    if block.redacted:
                        if is_same_model:
                            transformed_content.append(block)
                        continue
                    # Same model: keep signed thinking blocks (needed for replay),
                    # even if the thinking text is empty (OpenAI encrypted reasoning).
                    if is_same_model and block.thinking_signature:
                        transformed_content.append(block)
                        continue
                    if not block.thinking or block.thinking.strip() == "":
                        continue
                    if is_same_model:
                        transformed_content.append(block)
                    else:
                        transformed_content.append(TextContent(text=block.thinking))
                    continue

                if block.type == "text":
                    transformed_content.append(TextContent(text=block.text))
                    continue

                if block.type == "toolCall":
                    tool_call: ToolCall = block  # type: ignore[assignment]
                    normalized_call = tool_call

                    if not is_same_model and tool_call.thought_signature:
                        normalized_call = tool_call.model_copy(update={"thought_signature": None})

                    if not is_same_model and normalize_tool_call_id:
                        normalized_id = normalize_tool_call_id(tool_call.id)
                        if normalized_id != tool_call.id:
                            tool_call_id_map[tool_call.id] = normalized_id
                            normalized_call = normalized_call.model_copy(update={"id": normalized_id})

                    transformed_content.append(normalized_call)
                    continue

                transformed_content.append(block)

            transformed.append(assistant_msg.model_copy(update={"content": transformed_content}))
            continue

        transformed.append(msg)

    # Second pass: insert synthetic empty tool results for orphaned tool calls.
    # System messages are transparent to tool-call accounting: one that lands
    # between a tool call and its results is held back and emitted after the
    # results, so it never causes a duplicate result for a call answered later.
    result: List[Message] = []
    pending_tool_calls: List[ToolCall] = []
    existing_tool_result_ids: set[str] = set()
    held_system_messages: List[Message] = []

    def close_pending_tool_calls() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tc in pending_tool_calls:
                if tc.id not in existing_tool_result_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content=[TextContent(text="No result provided")],
                            is_error=True,
                            timestamp=int(time.time() * 1000),
                        )
                    )
            pending_tool_calls = []
            existing_tool_result_ids = set()
        result.extend(held_system_messages)
        held_system_messages.clear()

    for msg in transformed:
        if msg.role == "assistant":
            close_pending_tool_calls()

            # Skip errored/aborted assistant messages entirely: replaying partial
            # turns can cause API errors; the model retries from the last valid state.
            assistant_msg: AssistantMessage = msg  # type: ignore[assignment]
            if assistant_msg.stop_reason in ("error", "aborted"):
                continue

            tool_calls = [b for b in assistant_msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(msg)
        elif msg.role == "toolResult":
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
        elif msg.role == "system":
            if pending_tool_calls:
                held_system_messages.append(msg)
            else:
                result.append(msg)
        elif msg.role == "user":
            # A new user turn interrupts tool flow: insert synthetic results for orphans.
            close_pending_tool_calls()
            result.append(msg)
        else:
            result.append(msg)

    close_pending_tool_calls()
    return result
