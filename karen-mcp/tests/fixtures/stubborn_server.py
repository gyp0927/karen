"""Answers initialize, spawns a grandchild that outlives stdin, and ignores stdin EOF.

SIGTERM is ignored on POSIX too, so only the transport's process-group kill can
end it. The grandchild gives up after 30 seconds on its own, so a regression
cannot leak a process for the rest of the session.
"""

import json
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict

GRANDCHILD = (
    "import signal, time\n"
    "try:\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "except (OSError, ValueError):\n"
    "    pass\n"
    "time.sleep(30)\n"
)


def _write(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _read_stdin() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            # EOF is not a shutdown signal for this server: it keeps running
            # until it is killed.
            return
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("method") != "initialize":
            continue
        _write(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "stubborn-fixture", "version": "1.0.0"},
                },
            }
        )


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (OSError, ValueError):
        pass
    grandchild = subprocess.Popen(
        [sys.executable, "-c", GRANDCHILD],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"grandchild {grandchild.pid}", file=sys.stderr, flush=True)
    threading.Thread(target=_read_stdin, daemon=True).start()
    # Not an infinite loop: a transport that fails to kill this server should
    # leave a process behind for a minute, not for the rest of the session.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        time.sleep(1)


if __name__ == "__main__":
    main()
