"""Sample tool. A working example — add your own tools as new files in this package.

Decorate a plain function with ``@tool`` and it becomes an agent tool; the package auto-collects it
via ``all_tools()``, which ``agent.py`` uses. The schema the model sees is derived from the
signature and this docstring.
"""

from datetime import datetime

from agent.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
