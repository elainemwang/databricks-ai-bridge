"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Unlike the session store (short-term transcript for one conversation), long-term memory is exposed to
the model as two tools — ``remember`` and ``recall`` — over the Databricks managed memory store's
``agents/v1`` entries API. Facts persist across conversations.

No framework tool objects: ``memory_tools()`` returns plain **Chat Completions tool schemas** (the
``{"type": "function", "function": {...}}`` dicts you pass to ``chat.completions.create(tools=...)``),
and ``run_memory_tool(name, arguments)`` executes one when the model calls it. Both return nothing /
empty when ``AGENT_MEMORY_STORE`` is unset, so the model never sees the tools when memory is
unconfigured. The hand-rolled loop concatenates these schemas with its local tools and routes tool
calls by name.

Memory entries are per-actor. This uses ``AGENT_MEMORY_ACTOR_ID`` (default ``agent``), giving the
agent one shared long-term memory; change ``_actor_id`` to scope per user if needed.
"""

from __future__ import annotations

import os
from typing import Any

from databricks_mason.runtime.workspace import workspace_client

_AGENTS_V1 = "/api/agents/v1"

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Persist a durable fact about the user in long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember."},
                    "topic": {"type": "string", "description": "A short topic to file it under."},
                },
                "required": ["fact", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search the user's long-term memory for facts relevant to the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search memory for."},
                },
                "required": ["query"],
            },
        },
    },
]


def _enabled() -> bool:
    return bool(os.getenv("AGENT_MEMORY_STORE"))


def _actor_id() -> str:
    return os.getenv("AGENT_MEMORY_ACTOR_ID", "agent")


def _store_path() -> str:
    return f"{_AGENTS_V1}/memory-stores/{os.environ['AGENT_MEMORY_STORE']}"


def _api():
    # Build the client lazily (needs workspace auth) so importing this module stays cheap.
    return workspace_client().api_client


def memory_tools() -> list[dict[str, Any]]:
    """The Chat Completions tool schemas for long-term memory when configured, else none."""
    return [dict(schema) for schema in _TOOL_SCHEMAS] if _enabled() else []


def run_memory_tool(name: str, arguments: dict[str, Any]) -> str | None:
    """Execute a memory tool the model called; return the tool result, or ``None`` if not ours."""
    if not _enabled():
        return None
    if name == "remember":
        return _remember(arguments["fact"], arguments["topic"])
    if name == "recall":
        return _recall(arguments["query"])
    return None


def _remember(fact: str, topic: str) -> str:
    _api().do(
        "POST",
        f"{_store_path()}/entries",
        body={"actor_id": _actor_id(), "path": f"/{topic}/{fact[:8]}.md", "content": fact},
    )
    return "stored"


def _recall(query: str) -> str:
    data = _api().do(
        "POST",
        f"{_store_path()}/entries:search",
        body={"actor_id": _actor_id(), "query": query, "limit": 5},
    )
    entries = data.get("managed_memory_entries") or []
    return "\n".join(f"- {e.get('content')}" for e in entries) or "No relevant memories."
