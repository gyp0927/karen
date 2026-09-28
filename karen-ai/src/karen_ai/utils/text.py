"""Text extraction/rendering helpers, mirroring pi-ai's utils/text.ts."""

from __future__ import annotations

from typing import List, Union

from ..types import SystemMessage, TextContent


def content_text(content: Union[str, List], separator: str = "\n") -> str:
    """Extract and join text from message content."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if getattr(block, "type", None) == "text")


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: content followed by sections."""
    parts = [content_text(message.content)]
    for text in (message.sections or {}).values():
        if text is not None:
            parts.append(text)
    return "\n\n".join(part for part in parts if len(part) > 0)


def render_system_message_update(message: SystemMessage) -> str:
    """Render a later system message for APIs that accept mid-conversation system messages.

    Section changes are framed by name so the model can relate them to the
    leading prompt. This framing is request-time only.
    """
    parts: list[str] = []
    text = content_text(message.content)
    if len(text) > 0:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        if value is None:
            parts.append(f'Removed system prompt section "{name}".')
        else:
            parts.append(f'Updated system prompt section "{name}":\n\n{value}')
    return "\n\n".join(parts)
