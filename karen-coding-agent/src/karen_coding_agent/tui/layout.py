"""Layout + ANSI renderer for the alt-screen chat viewport (pure logic).

Mirrors pi's `chat-viewport.ts`: a scrollable transcript region on top and a
fixed input dock below (status line, multi-line editor, footer). Everything
here is terminal-agnostic and testable: given a `Transcript` and terminal
dimensions it produces a list of ANSI lines, and the thin terminal layer in
`terminal.py` writes them.

Row accounting is exact: the dock is rendered first and the transcript gets
the remaining rows, so `render` always returns exactly `height` lines and no
dock row is ever silently dropped.

ANSI colors are applied only when `color` is on (the controller passes
`sys.stdout.isatty()` — piped output stays plain for tests and greps).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .transcript import TMessage, Transcript

_RESET = "\x1b[0m"


class _Style:
    """Tiny SGR helper so message styles read like pi's theme calls."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def fg(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}{_RESET}" if self.color else text

    def bg(self, code: str, text: str) -> str:
        return f"\x1b[48;5;{code}m{text}{_RESET}" if self.color else text


@dataclass
class _LayoutInput:
    """What the dock and the status line need to know about the app."""

    status: str = ""
    editor_lines: Tuple[str, ...] = ()
    footer: str = ""
    #: Text queued by a branch move, shown in the editor when it is empty.
    pending_input: str = ""


class Layout:
    def __init__(self, width: int, height: int, color: bool = True) -> None:
        self.width = max(20, width)
        self.height = max(4, height)
        self.style = _Style(color)
        self.follow = True  # stick to the bottom until the user scrolls up
        self.top = 0

    def resize(self, width: int, height: int) -> None:
        self.width = max(20, width)
        self.height = max(4, height)
        # No clamping here: a resize cannot know the transcript length, and
        # `render` clamps against it. Clamping to the screen height would
        # teleport a deeply-scrolled viewport to the top of the buffer.

    # -- transcript rendering -------------------------------------------------

    def _wrap(self, text: str, width: int, prefix: str = "", prefix_width: int = 0) -> List[str]:
        """Wrap paragraph text to `width` columns; each line carries `prefix`.

        `prefix_width` is the prefix's *visible* width, which differs from
        `len(prefix)` when the prefix carries ANSI codes.
        """
        avail = max(1, width - prefix_width)
        if not text:
            return [prefix]
        lines: List[str] = []
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append(prefix)
                continue
            chunks = textwrap.wrap(paragraph, width=avail) or [paragraph]
            lines.extend(prefix + chunk[:avail] for chunk in chunks)
        return lines

    def _tool_lines(self, message: TMessage, width: int) -> List[str]:
        tool = message.tool
        state = {
            "running": self.style.fg("36", f"[tool ->] {tool.name}"),
            "done": self.style.fg("38;5;245", f"[tool <-] {tool.name}"),
            "error": self.style.fg("31", f"[tool <-] {tool.name} ERROR"),
        }[tool.state]
        lines = [state]
        args = tool.args
        if isinstance(args, dict) and args:
            preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
            if len(preview) > 72:
                preview = preview[:71] + "…"
            lines.extend(self._wrap(preview, width, "    ", 4))
        if tool.state == "error" and tool.error_text:
            for line in tool.error_text.splitlines()[:10]:
                lines.extend(self._wrap(line, width, "    ", 4))
        return lines

    def _message_lines(self, message: TMessage, width: int) -> List[str]:
        s = self.style
        if message.kind == "user":
            marker_plain = "❯ "
            marker = marker_plain if not message.user.pending else s.fg("38;5;245", marker_plain)
            if not message.user.pending:
                return self._wrap(message.user.text, width, marker, len(marker_plain))
            # Reserve room for the " (pending)" suffix inside the wrap, then
            # append it to the last row: the extra text is part of the visible
            # line and must not push the row past the terminal width.
            suffix = " (pending)"
            lines = self._wrap(message.user.text, width - len(suffix), marker, len(marker_plain))
            lines[-1] = lines[-1] + s.fg("38;5;245", suffix)
            return lines
        if message.kind == "assistant":
            if message.assistant.streaming and not message.assistant.text.strip():
                return [s.fg("90", "… thinking")]
            lines = self._wrap(message.assistant.text, width)
            if message.assistant.streaming and lines:
                lines[-1] = lines[-1] + s.fg("32", "▌")
            if message.assistant.error:
                lines.extend(self._wrap(f"run failed: {message.assistant.error}", width))
            return lines
        if message.kind == "tool":
            return self._tool_lines(message, width)
        notice = message.notice
        prefix = s.fg("31", "! ") if notice.error else s.fg("38;5;245", "· ")
        return self._wrap(notice.text, width, prefix, 2)

    def transcript_lines(self, transcript: Transcript) -> List[str]:
        width = self.width
        lines: List[str] = []
        for message in transcript.messages:
            lines.extend(self._message_lines(message, width))
            lines.append("")
        # pi's chat viewport hides the trailing blank so the last block sits
        # flush against the separator, not one row above it.
        if lines:
            lines.pop()
        return lines

    # -- the fixed dock --------------------------------------------------------

    def _status_line(self, app: _LayoutInput) -> str:
        text = self._truncate(app.status, self.width)
        return text.ljust(self.width)

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if len(text) > width:
            return text[: max(0, width - 1)] + "…"
        return text

    def _editor_rows(self, app: _LayoutInput) -> List[str]:
        rows = list(app.editor_lines) or [app.pending_input]
        if not rows:
            rows = [""]
        return rows

    def _editor_lines(self, app: _LayoutInput) -> List[str]:
        s = self.style
        rows = self._editor_rows(app)
        prompt = s.fg("32", "> ")
        out = []
        for index, line in enumerate(rows):
            prefix = prompt if index == 0 else "  "
            # Keep every editor row within the terminal width: a long typed
            # line must not wrap into extra physical rows the dock did not
            # reserve (that would desynchronise the frame). Reserve 2 columns
            # for the prompt and 1 for the cursor block.
            body = self._truncate(line, max(1, self.width - 3))
            suffix = s.fg("32", "▌") if index == len(rows) - 1 else ""
            out.append(f"{prefix}{body}{suffix}")
        return out

    def _dock_lines(self, app: _LayoutInput) -> List[str]:
        s = self.style
        lines = [s.bg("238", self._status_line(app))]
        lines.extend(self._editor_lines(app))
        lines.append(s.fg("38;5;245", self._truncate(app.footer, self.width)))
        return lines

    # -- composition -----------------------------------------------------------

    def render(self, transcript: Transcript, app: Optional[_LayoutInput] = None, **_) -> List[str]:
        """Produce exactly `self.height` lines (ANSI strings).

        The dock is rendered first and measured; the transcript fills the rows
        that remain and is scrolled/padded to fit. `follow` sticks the view to
        the newest line, otherwise the user's scroll offset (`self.top`) is
        honored.
        """
        app = app or _LayoutInput()
        dock = self._dock_lines(app)
        region = max(1, self.height - len(dock))
        all_lines = self.transcript_lines(transcript)
        total = len(all_lines)
        if self.follow:
            self.top = max(0, total - region)
        self.top = max(0, min(self.top, max(0, total - region)))
        visible = all_lines[self.top : self.top + region]
        visible = list(visible) + [""] * (region - len(visible))
        out = visible + dock
        # The dock may be taller than the screen (a very small terminal): keep
        # the *last* height rows so the footer stays visible.
        if len(out) > self.height:
            out = out[len(out) - self.height :]
        return out

    # -- scrolling -------------------------------------------------------------

    def scroll_up(self, lines: int) -> None:
        self.follow = False
        self.top = max(0, self.top - lines)

    def scroll_down(self, lines: int) -> None:
        self.top += lines
        # A big scroll (the "jump to bottom" affordance) re-locks follow; a
        # normal step just moves the window. `top` is clamped in render().
        if lines >= self.height:
            self.follow = True
            self.top = 0

    def jump_to_bottom(self) -> None:
        self.follow = True
        self.top = 0
