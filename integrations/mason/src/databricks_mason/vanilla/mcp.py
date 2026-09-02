"""Hand-rolled MCP integration for the vanilla agent — the raw wire protocol, no framework.

``mcp_toolset()`` reads the MCP servers declared in ``agent.toml`` (sandbox/mcp + uc_function),
opens a streamable-HTTP MCP session to each with Databricks auth, and returns a ``McpToolset``: the
tools' **Chat Completions schemas** (``{"type": "function", ...}`` dicts, ready for
``chat.completions.create(tools=...)``) plus a ``call(name, arguments)`` dispatcher that runs the tool
on the right server (applying sandbox downscoping via the MCP call's ``_meta``).

The toolset is an async context manager: open it for the life of a request so the sessions stay
connected while the agent's loop calls tools. Fail-open — if a server can't be reached, its tools are
simply absent rather than crashing the turn.

This uses the ``mcp`` client library directly (``streamablehttp_client`` + ``ClientSession``); there
is no agent SDK in the loop, mirroring how the model is called through a raw OpenAI client.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession

try:  # mcp >= 2.0 renamed the streamable-HTTP client; support both spellings.
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - older mcp
    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

from databricks_mason.runtime.tool_manifest import ToolRecord, downscope_wire, load_tools
from databricks_mason.runtime.workspace import workspace_client, workspace_headers

logger = logging.getLogger(__name__)

_FRAMEWORK = "vanilla"


class _DatabricksBearerAuth(httpx.Auth):
    """Attach the SDK's current Authorization header to each MCP request (refreshes per call)."""

    def __init__(self, authenticate) -> None:
        self._authenticate = authenticate

    def auth_flow(self, request):
        request.headers["Authorization"] = self._authenticate()["Authorization"]
        yield request


def _server_url(tool: ToolRecord, host: str) -> str | None:
    if tool.kind in {"sandbox", "mcp"}:
        return f"{host}/ai-gateway/mcp-services/{tool.service}"
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return f"{host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
    return None


class McpToolset:
    """Connected MCP sessions plus the Chat Completions schemas for their tools.

    Built by :func:`mcp_toolset` as an async context manager. ``schemas`` are the tool definitions to
    pass to the model; ``call`` dispatches a model tool call to the session that owns it. Sandbox
    tools carry their manifest downscope, applied as the MCP call's ``_meta``.
    """

    def __init__(self) -> None:
        self._schemas: list[dict[str, Any]] = []
        # tool name -> (session, downscope|None)
        self._by_name: dict[str, tuple[ClientSession, dict[str, Any] | None]] = {}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """The Chat Completions tool schemas for every reachable MCP tool."""
        return self._schemas

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run MCP tool ``name`` with ``arguments`` and return its text content."""
        session, downscope = self._by_name[name]
        meta = {"downscope": downscope} if downscope else None
        result = await session.call_tool(name, arguments, meta=meta)
        texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        return "\n".join(texts)

    def handles(self, name: str) -> bool:
        return name in self._by_name


async def mcp_toolset(stack: AsyncExitStack, extra_urls: list[str] | None = None) -> McpToolset:
    """Connect the agent's MCP servers and collect their tools. Fail-open per server.

    Opens a session to each server declared in ``agent.toml`` (plus any URLs in ``extra_urls``) on the
    provided ``AsyncExitStack``, so they stay connected until the stack closes. Returns a
    :class:`McpToolset` with the reachable tools' schemas and a dispatcher. A server that can't be
    reached or listed is skipped with a warning rather than failing the request.
    """
    toolset = McpToolset()
    client = workspace_client()
    host = client.config.host.rstrip("/")
    auth = _DatabricksBearerAuth(client.config.authenticate)
    headers = workspace_headers() or None

    try:
        tools = load_tools(expected_framework=_FRAMEWORK)
    except Exception:
        logger.warning("Could not load agent.toml tools; continuing without MCP.", exc_info=True)
        tools = ()

    declared = [(tool, _server_url(tool, host)) for tool in tools]
    targets: list[tuple[str, dict[str, Any] | None]] = [
        (url, downscope_wire(tool) if tool.kind == "sandbox" else None)
        for tool, url in declared
        if url is not None
    ]
    targets += [(url, None) for url in (extra_urls or [])]

    for url, downscope in targets:
        try:
            await _connect(stack, toolset, url, downscope, auth, headers)
        except Exception:
            logger.warning("MCP server %s unavailable; continuing without it.", url, exc_info=True)
    return toolset


async def _connect(
    stack: AsyncExitStack,
    toolset: McpToolset,
    url: str,
    downscope: dict[str, Any] | None,
    auth: httpx.Auth,
    headers: dict[str, str] | None,
) -> None:
    read, write, _ = await stack.enter_async_context(
        streamable_http_client(url, headers=headers, auth=auth, timeout=120)
    )
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    for tool in (await session.list_tools()).tools:
        toolset._schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            }
        )
        toolset._by_name[tool.name] = (session, downscope)
