"""Application-level tools for the karen coding agent (pi coding-agent's
`core/tools/`): find, grep, ls (in-process ports of pi's fd/rg wrappers) and
powershell (Windows only), layered on karen-agent's read/write/edit/bash.

`create_default_tools(cwd)` mirrors pi's default tool set and order:
read, bash, powershell (Windows), edit, write, grep, find, ls.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from karen_agent.tools import (
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from karen_agent.types import AgentTool

from .find import FIND_DESCRIPTION, FIND_SCHEMA, create_find_tool
from .grep import GREP_DESCRIPTION, GREP_SCHEMA, create_grep_tool
from .ls import LS_DESCRIPTION, LS_SCHEMA, create_ls_tool


def create_default_tools(cwd: Optional[str] = None) -> List[AgentTool]:
    """karen's default tool set, in pi's order (powershell only on Windows)."""
    tools: List[AgentTool] = [create_read_tool(cwd), create_bash_tool(cwd)]
    if sys.platform == "win32":
        from .powershell import create_powershell_tool

        tools.append(create_powershell_tool(cwd))
    tools.extend(
        [
            create_edit_tool(cwd),
            create_write_tool(cwd),
            create_grep_tool(cwd),
            create_find_tool(cwd),
            create_ls_tool(cwd),
        ]
    )
    return tools


__all__ = [
    "FIND_DESCRIPTION",
    "FIND_SCHEMA",
    "GREP_DESCRIPTION",
    "GREP_SCHEMA",
    "LS_DESCRIPTION",
    "LS_SCHEMA",
    "create_default_tools",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
]
