"""Alt-screen TUI package (hand-rolled, pure stdlib).

Layout: a scrollable transcript on top and a fixed input dock below
(status / multi-line editor / footer), mirroring pi's chat-viewport.
The logic is split so it is testable without a TTY:

  transcript.py  — pure event -> message model
  layout.py      — pure message + terminal dims -> ANSI lines
  terminal.py    — thin alt-screen / raw-key controller
  app.py         — the loop that wires them together
"""

from .app import LiveTui, TuiApp
from .layout import Layout
from .terminal import TerminalController
from .transcript import TMessage, Transcript

__all__ = ["LiveTui", "TuiApp", "Layout", "TerminalController", "TMessage", "Transcript"]
