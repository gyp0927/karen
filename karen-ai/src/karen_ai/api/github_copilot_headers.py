"""GitHub Copilot dynamic headers, mirroring api/github-copilot-headers.ts."""

from __future__ import annotations

from typing import Dict, List, Sequence


def infer_copilot_initiator(messages: Sequence) -> str:
    """Copilot expects X-Initiator to indicate whether the request is
    user-initiated or agent-initiated (follow-up after assistant/tool messages)."""
    last = messages[-1] if messages else None
    return "agent" if last is not None and last.role != "user" else "user"


def has_copilot_vision_input(messages: Sequence) -> bool:
    """Copilot requires the Copilot-Vision-Request header when sending images."""
    for msg in messages:
        if msg.role in ("user", "toolResult") and isinstance(msg.content, list):
            if any(getattr(c, "type", None) == "image" for c in msg.content):
                return True
    return False


def build_copilot_dynamic_headers(messages: List, has_images: bool) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
