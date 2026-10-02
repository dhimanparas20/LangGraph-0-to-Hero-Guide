"""File management tools - a sandboxed workspace directory for the agent.

FileManagementToolkit (langchain-community) gives the agent seven tools -
copy_file, file_delete, file_search, move_file, read_file, write_file,
list_directory - that operate INSIDE the datastore/ directory only: the
toolkit's root_dir sandboxes every path, so the agent can never touch
files outside it.

The directory defaults to <repo>/datastore; override with DATASTORE_DIR
in .env. The directory is created on load if missing.

Note: langchain-community is in maintenance mode (sunset warning on
import). Everything still works; if it is ever removed, these seven
tools are simple to hand-write with plain pathlib - lesson 4 taught
exactly that pattern.
"""

import os
from pathlib import Path

# sandbox root: DATASTORE_DIR env var, else <repo>/datastore
DATASTORE_DIR = Path(os.getenv("DATASTORE_DIR") or (Path(__file__).parent.parent.parent / "datastore"))


def get_datastore_dir() -> Path:
    """Returns the datastore directory, creating it if needed.

    Returns:
        The resolved Path to the agent's file sandbox.
    """
    DATASTORE_DIR.mkdir(parents=True, exist_ok=True)
    return DATASTORE_DIR


def load_file_management_tools() -> list:
    """Loads the sandboxed file management tools for the agent.

    Returns:
        A list of seven LangChain tools (write_file, read_file,
        list_directory, file_search, move_file, copy_file, file_delete),
        all restricted to the datastore directory.
    """
    from langchain_community.agent_toolkits import FileManagementToolkit

    root = get_datastore_dir()
    toolkit = FileManagementToolkit(root_dir=str(root))
    tools = toolkit.get_tools()
    print("[files] sandbox at " + str(root) + " -> " + str([t.name for t in tools]))
    return tools
