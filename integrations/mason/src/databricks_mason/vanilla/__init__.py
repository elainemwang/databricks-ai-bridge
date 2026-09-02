"""Vanilla (hand-rolled) adapter for running an agent on Databricks (installed via ``databricks-mason[runtime-vanilla]``).

No agent framework — the agent is a plain loop over the **Chat Completions** API that you write
yourself (see ``agent/agent.py`` in the template). This adapter supplies only the Databricks-shaped
pieces that loop needs, each usable on its own:

- ``chat_client()`` — a raw ``openai.AsyncOpenAI`` pointed at the workspace's ``/serving-endpoints``
  with Databricks auth wired in. Call it however you like; nothing wraps your loop.
- ``session_store(session_id)`` — a message-list store (in-process by default; durable via the
  managed Session Store when ``AGENT_SESSION_STORE`` is set).
- ``mcp_toolset(...)`` — fetch tool schemas from the MCP servers declared in ``agent.toml`` and call
  them, over the raw MCP wire protocol (no framework).
- ``memory_tools()`` / ``run_memory_tool(...)`` — long-term ``remember``/``recall`` as plain Chat
  Completions tool schemas plus a dispatcher.
- ``configure_tracing()`` — MLflow tracing (OpenAI autolog), opt-in via env.

Everything is Chat-Completions-shaped plain dicts, so you can read exactly what the agent sends and
receives. ``__all__`` is the curated surface; other entry points are reachable by submodule path.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.runtime import tag_session, workspace_client, workspace_headers
    from databricks_mason.vanilla.mcp import mcp_toolset
    from databricks_mason.vanilla.memory import memory_tools, run_memory_tool
    from databricks_mason.vanilla.model import chat_client
    from databricks_mason.vanilla.sessions import session_store


def configure_tracing() -> None:
    """Enable MLflow tracing with OpenAI autologging. Call once at startup.

    Safe to call unconditionally — tracing turns on only when the MLflow destination and experiment
    are configured in the environment (see :func:`databricks_mason.runtime.configure_tracing`). The
    hand-rolled loop calls Chat Completions through a raw OpenAI client, so OpenAI autolog captures it.
    """
    import mlflow

    from databricks_mason.runtime import configure_tracing as _configure_tracing

    _configure_tracing(autolog=mlflow.openai.autolog)


__all__ = [
    # Raw Chat Completions client, Databricks-authed — the agent calls it directly.
    "chat_client",
    # MCP tool schemas + calls from agent.toml (plus any you pass), over the raw MCP protocol.
    "mcp_toolset",
    # Long-term memory tools (opt-in via AGENT_MEMORY_STORE) — Chat Completions tool schemas.
    "memory_tools",
    "run_memory_tool",
    # Conversation persistence — a message-list store, in-process or durable.
    "session_store",
    # MLflow tracing (OpenAI autolog bound in) — call configure_tracing() once at startup.
    "configure_tracing",
    "tag_session",
    # Workspace SDK client construction.
    "workspace_client",
    "workspace_headers",
]

# Re-exports resolved lazily (PEP 562) so importing one submodule (e.g. ``.sessions``) does not
# eagerly pull in the others' dependencies. ``configure_tracing`` is defined above (binds OpenAI autolog).
_MODULE_BY_NAME = {
    "chat_client": "databricks_mason.vanilla.model",
    "mcp_toolset": "databricks_mason.vanilla.mcp",
    "memory_tools": "databricks_mason.vanilla.memory",
    "run_memory_tool": "databricks_mason.vanilla.memory",
    "session_store": "databricks_mason.vanilla.sessions",
    "tag_session": "databricks_mason.runtime",
    "workspace_client": "databricks_mason.runtime",
    "workspace_headers": "databricks_mason.runtime",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
