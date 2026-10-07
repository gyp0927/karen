"""A newline-delimited MCP server over stdio (pi's `test/fixtures/stdio-server.mjs`).

Everything a stdio test needs: an `echo` tool, a `ping`, and a `noise` tool that
writes an unparseable line to stdout before answering it (pi's suite covers
stray output through `emitError` directly; here it is worth a real pipe).
"""

import json
import sys
from typing import Any, Dict


def _write(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _handle(message: Dict[str, Any]) -> Dict[str, Any]:
    method = message.get("method")
    if method == "initialize":
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stdio-fixture", "version": "1.0.0"},
        }
    if method == "tools/list":
        return {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") == "noise":
            # Not JSON: the client has to report it and carry on.
            sys.stdout.write("this is not json\n")
            sys.stdout.flush()
            return {"content": [{"type": "text", "text": "noisy"}]}
        return {"content": [{"type": "text", "text": str((params.get("arguments") or {}).get("text"))}]}
    if method == "ping":
        return {}
    raise ValueError(f"Method not found: {method}")


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8")
    print("stdio fixture ready", file=sys.stderr, flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        message = json.loads(line)
        if "id" not in message or "method" not in message:
            continue
        try:
            result = _handle(message)
        except BaseException as error:
            _write({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": str(error)}})
            continue
        _write({"jsonrpc": "2.0", "id": message["id"], "result": result})


if __name__ == "__main__":
    main()
