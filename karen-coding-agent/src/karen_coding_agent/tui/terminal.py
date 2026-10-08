"""Thin terminal controller: alternate screen buffer, raw mode, key reads.

This is the only module that touches the real TTY. Everything else in the
TUI package is pure logic. Raw mode is enabled on entry (POSIX cbreak via
termios; Windows console VT input via msvcrt + SetConsoleMode) and restored
on exit, so individual keypresses arrive without line buffering or echo.

Alt screen: CSI ?1049 h/l. Clear + hide cursor on entry, restore on exit.
"""

from __future__ import annotations

import sys
from typing import Optional


def _is_console_handle(handle) -> bool:
    """True when `handle` is a real Windows console.

    `isatty()` cannot answer this: the NUL device is a character device, so a
    process with `karen < NUL > NUL` (a scheduled or console-less launch)
    reports True on both ends. `GetConsoleMode` is the same call the raw-mode
    setup below depends on, so it is the honest gate.
    """
    try:
        import ctypes

        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except Exception:
        return False


def _has_windows_console(std_handle: int) -> bool:
    """`std_handle`: -10 = STD_INPUT_HANDLE, -11 = STD_OUTPUT_HANDLE."""
    try:
        import ctypes

        return _is_console_handle(ctypes.windll.kernel32.GetStdHandle(std_handle))
    except Exception:
        return False


def supports_tty() -> bool:
    """True when this process has a usable interactive terminal on both ends.

    Both streams are required: the alt-screen escapes go to stdout, so a
    redirected stdout would dump them into a file, and the raw-mode/key path
    needs a real stdin. On Windows that means a real *console*, not just
    something `isatty()` likes — `msvcrt.getwch()` has no EOF path, so a
    NUL-backed stdin would park the key loop forever instead of ending it, and
    Git Bash (mintty) hands a native program pipes rather than a console.

    Failing this check is not fatal: the caller falls back to the plain REPL.
    """
    if not (
        hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    ):
        return False
    if sys.platform == "win32":
        return _has_windows_console(-10) and _has_windows_console(-11)
    return True


class TerminalController:
    """Enters/leaves the alt screen and hands raw key bytes to the app loop."""

    def __init__(self, in_stream, out_stream) -> None:
        self._in = in_stream
        self._out = out_stream
        self._in_alt_screen = False
        # POSIX raw-mode bookkeeping.
        self._termios_mod = None  # the `termios` module, once saved
        self._saved_termios: Optional[object] = None
        # Windows console-mode bookkeeping.
        self._win32 = sys.platform == "win32"
        self._saved_console_mode: Optional[int] = None
        self._console_handle = None

    # -- lifecycle -------------------------------------------------------------

    def enter(self) -> None:
        if self._in_alt_screen:
            return
        self._enter_raw_mode()
        # Enter alt screen (1049 also clears + hides the cursor).
        self._write("\x1b[?1049h")
        # Hide the hardware cursor; the editor draws its own block.
        self._write("\x1b[?25l")
        self._in_alt_screen = True

    def leave(self) -> None:
        if not self._in_alt_screen:
            return
        # Restore cursor and return to the main screen (1049l restores the
        # saved screen, cursor position and visibility).
        self._write("\x1b[?25h")
        self._write("\x1b[?1049l")
        self._in_alt_screen = False
        self._exit_raw_mode()

    def _write(self, data: str) -> None:
        self._out.write(data)
        try:
            self._out.flush()
        except Exception:
            pass

    def write(self, data: str) -> None:
        """Public sink used by `TuiApp` to paint a frame."""
        self._write(data)

    # -- raw mode ---------------------------------------------------------------

    def _enter_raw_mode(self) -> None:
        if self._win32:
            self._enter_windows_raw_mode()
        else:
            self._enter_posix_raw_mode()

    def _exit_raw_mode(self) -> None:
        if self._win32:
            self._exit_windows_raw_mode()
        else:
            self._exit_posix_raw_mode()

    # POSIX: cbreak mode (char-at-a-time, no echo), restored on exit.
    def _enter_posix_raw_mode(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._saved_termios = termios.tcgetattr(fd)
            self._termios_mod = termios
            tty.setcbreak(fd)
        except Exception:
            self._termios_mod = None
            self._saved_termios = None

    def _exit_posix_raw_mode(self) -> None:
        if self._termios_mod is None or self._saved_termios is None:
            return
        try:
            self._termios_mod.tcsetattr(
                sys.stdin.fileno(), self._termios_mod.TCSADRAIN, self._saved_termios
            )
        except Exception:
            pass
        finally:
            self._termios_mod = None
            self._saved_termios = None

    # Windows: enable VT processing + raw console input on the real handles.
    def _enter_windows_raw_mode(self) -> None:
        try:
            import ctypes
            import msvcrt  # noqa: F401  (used in _read_byte_windows)

            kernel32 = ctypes.windll.kernel32
            # STD_INPUT_HANDLE = -10; STD_OUTPUT_HANDLE = -11.
            self._console_handle = kernel32.GetStdHandle(-10)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(self._console_handle, ctypes.byref(mode)):
                self._saved_console_mode = mode.value
                # ENABLE_PROCESSED_INPUT(0x1) off, ENABLE_LINE_INPUT(0x2) off,
                # ENABLE_ECHO_INPUT(0x4) off -> raw console input.
                raw = mode.value & ~0x0001 & ~0x0002 & ~0x0004
                kernel32.SetConsoleMode(self._console_handle, raw)
            out_handle = kernel32.GetStdHandle(-11)
            out_mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(out_handle, ctypes.byref(out_mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004) so VT escapes work.
                kernel32.SetConsoleMode(out_handle, out_mode.value | 0x0004)
        except Exception:
            self._console_handle = None
            self._saved_console_mode = None

    def _exit_windows_raw_mode(self) -> None:
        if self._console_handle is None or self._saved_console_mode is None:
            return
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleMode(self._console_handle, self._saved_console_mode)
        except Exception:
            pass
        finally:
            self._console_handle = None
            self._saved_console_mode = None

    # -- key reads ----------------------------------------------------------------

    def read_key(self) -> Optional[str]:
        """Read one key (or one byte of an escape sequence).

        Returns a symbolic key name ("up", "ctrl_c", "enter", …) or a literal
        character for printable input. Returns None on EOF. Blocking read: the
        app loop drives this, so it is only called when a key is due.
        """
        byte = self._read_byte()
        if byte is None:
            return None
        if byte == 0x1B:  # ESC
            return self._read_escape()
        return self._decode_byte(byte)

    def _decode_byte(self, byte: int) -> str:
        # ASCII control codes.
        if byte == 3:
            return "ctrl_c"
        if byte == 4:
            return "ctrl_d"
        if byte == 9:
            return "tab"
        if byte == 10:
            return "enter"
        if byte == 13:
            return "enter"
        if byte in (0x7F, 8):  # DEL or Ctrl-H
            return "backspace"
        # Ctrl+letter maps to 1..26.
        if 1 <= byte <= 26:
            return f"ctrl_{chr(96 + byte)}"
        return chr(byte)

    def _read_escape(self) -> str:
        """ESC was read. Disambiguate: lone Escape, CSI (ESC [), SS3 (ESC O),
        or Alt+<char>."""
        nxt = self._read_byte()
        if nxt is None:
            return "escape"
        if nxt == 0x5B:  # '[' -> CSI
            return self._read_csi()
        if nxt == 0x4F:  # 'O' -> SS3
            third = self._read_byte()
            return _SS3_KEYS.get(third, "escape") if third is not None else "escape"
        # Alt+<char>
        return f"alt_{chr(nxt).lower()}"

    def _read_csi(self) -> str:
        """Read the rest of a CSI sequence (`[` already consumed).

        The payload is digits/`;` up to the final byte (a letter in 0x40-0x7E
        or `~`). The final byte is kept, so the code looks like "A" (arrow),
        "1;5C" (modified arrow), or "24~" (F-key).
        """
        buf = []
        for _ in range(16):
            b = self._read_byte()
            if b is None:
                break
            ch = chr(b)
            buf.append(ch)
            if 0x40 <= b <= 0x7E:  # reached the final byte
                break
        return _decode_csi("".join(buf))

    # -- raw byte reads ---------------------------------------------------------

    def _read_byte(self) -> Optional[int]:
        if self._win32:
            return self._read_byte_windows()
        return self._read_byte_stream()

    def _read_byte_stream(self) -> Optional[int]:
        try:
            data = self._in.read(1)
        except Exception:
            return None
        if not data:
            return None
        if isinstance(data, str):
            return ord(data)
        return data[0]

    def _read_byte_windows(self) -> Optional[int]:
        try:
            import msvcrt

            ch = msvcrt.getwch()  # blocks for one char, no echo in raw mode
        except Exception:
            return None
        if ch in ("\x00", "\xe0"):
            # Windows extended keys (arrows/F-keys) arrive as a 0/0xE0 prefix
            # followed by a scan code; translate the common ones.
            scan = msvcrt.getwch()
            return _WIN32_SCAN.get(ord(scan), None)
        return ord(ch)

    @staticmethod
    def terminal_size() -> tuple:
        """(rows, cols) of the real terminal; falls back to 24x80."""
        try:
            import shutil

            cols, rows = shutil.get_terminal_size((80, 24))
            return rows, cols
        except Exception:
            return 24, 80


# Windows conhost scan codes for extended keys (after 0x00/0xE0 prefix).
# The read loop turns these into the same symbolic names as the POSIX path.
_WIN32_SCAN = {
    72: "up",
    80: "down",
    75: "left",
    77: "right",
    73: "pageup",
    81: "pagedown",
    71: "home",
    79: "end",
    82: "insert",
    83: "delete",
}

# SS3 (ESC O) key codes.
_SS3_KEYS = {
    0x41: "up",
    0x42: "down",
    0x43: "right",
    0x44: "left",
    0x48: "home",
    0x46: "end",
}


def _decode_csi(code: str) -> str:
    # code is everything after `[`, e.g. "A", "1;5C", "24~".
    if not code:
        return "escape"

    # Modified arrows and keys: "<param>;<modifier><letter>".
    if ";" in code:
        parts = code.split(";")
        final = parts[-1]
        letter = final[-1]
        if letter in "ABCD":
            base = {"A": "up", "B": "down", "C": "right", "D": "left"}[letter]
            modifier = final[:-1] or "1"
            return _modifier_prefix(modifier) + base
        if letter == "~" and len(parts) >= 1:
            return _decode_tilde(parts[-2] if len(parts) > 1 else parts[0])
        return "unknown"

    # Plain arrows / Home / End.
    if code in ("A",):
        return "up"
    if code in ("B",):
        return "down"
    if code in ("C",):
        return "right"
    if code in ("D",):
        return "left"
    if code in ("H",):
        return "home"
    if code in ("F",):
        return "end"
    if code == "Z":
        return "shift_tab"

    # Tilde-terminated keys (Delete/Insert/PgUp/PgDn/F-keys).
    if code.endswith("~"):
        return _decode_tilde(code[:-1])

    if code == "M":
        return "mouse"
    return "unknown"


def _decode_tilde(param: str) -> str:
    table = {
        "1": "home",
        "2": "insert",
        "3": "delete",
        "4": "end",
        "5": "pageup",
        "6": "pagedown",
        "7": "home",
        "8": "end",
        "11": "f1",
        "12": "f2",
        "13": "f3",
        "14": "f4",
        "15": "f5",
        "17": "f6",
        "18": "f7",
        "19": "f8",
        "20": "f9",
        "21": "f10",
        "23": "f11",
        "24": "f12",
    }
    return table.get(param, "unknown")


def _modifier_prefix(modifier: str) -> str:
    # xterm modifiers: 2=Shift, 3=Alt, 4=Shift+Alt, 5=Ctrl, 6=Shift+Ctrl, ...
    mapping = {
        "2": "shift_",
        "3": "alt_",
        "4": "shift_alt_",
        "5": "ctrl_",
        "6": "shift_ctrl_",
        "7": "alt_ctrl_",
        "8": "shift_alt_ctrl_",
    }
    return mapping.get(modifier, "")
