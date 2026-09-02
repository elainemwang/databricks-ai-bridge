"""The agent — a hand-rolled loop over the Chat Completions API. No agent framework.

This is the whole reasoning plane: ``stream_handler`` runs the classic tool-calling loop yourself —
call the model, if it asks for tools run them and feed the results back, repeat until it answers —
streaming each step as it happens. ``invoke_handler`` is the same run collected into one response.

Everything Databricks-shaped comes from ``databricks_mason.vanilla`` as plain pieces you wire in: a
raw OpenAI client (``chat_client``), a message-list session store (``session_store``), MCP tools over
the wire protocol (``mcp_toolset``), and long-term memory tool schemas (``memory_tools`` /
``run_memory_tool``). Nothing wraps the loop — read it top to bottom to see exactly what an agent is.

Messages are Chat Completions dicts throughout (``{"role", "content", "tool_calls"}``), so the store,
the model call, and the browser UI all speak the same shape with no translation.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

from databricks_mason import tag_session, workspace_client
from databricks_mason.vanilla import (
    chat_client,
    configure_tracing,
    mcp_toolset,
    memory_tools,
    run_memory_tool,
    session_store,
)

from agent.mcps import build_mcp_urls

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"
SYSTEM_PROMPT = "You are a helpful assistant."

# Cap the tool-calling loop so a misbehaving model can't spin forever. Each pass is one model call
# plus any tools it requested; raise it for agents that legitimately chain many tool calls.
MAX_STEPS = 8


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    configure_tracing()


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable.
    """
    try:
        workspace_client()
    except Exception as e:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = (
            f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
        )
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
            "Fix one of:\n"
            "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
            "  • run `databricks auth login --profile <name>` to create one, or\n"
            "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
            f"(underlying error: {e})"
        ) from e


def _session_id(request: dict) -> str:
    """Return the session id derived by the runtime from the Apps routing cookie.

    Clients do not send ``session_id`` in the body. The runtime makes the cookie value available to
    the handler after resolving the deployed Apps cookie or the local-development fallback cookie.
    """
    return str(request["session_id"])


async def invoke_handler(request: dict) -> dict:
    """Run one turn to completion. Called by the runtime for POST /invocations.

    ``request`` is a dict with an ``input`` list of Chat Completions message dicts; the returned dict
    carries the run's new messages (same shape) and the ``session_id`` to pass back next turn.
    """
    request = {**request, "session_id": _session_id(request)}
    outputs = [event["message"] async for event in stream_handler(request) if event["type"] == "message"]
    return {
        "output": outputs,
        "session_id": request["session_id"],
        "status": "completed",
    }


async def stream_handler(request: dict) -> AsyncGenerator[dict, None]:
    """Run the hand-rolled tool-calling loop, streaming each step. Called by the runtime for stream=true.

    Emits the runtime's SDK-agnostic envelope — ``{"type": "delta", ...}`` for streamed assistant
    text, ``{"type": "message", "message": {...}}`` for each completed message (assistant turns, tool
    calls, tool results) — the same shape every mason template produces, so the runtime and browser UI
    are identical across frameworks.
    """
    session_id = _session_id(request)
    tag_session(session_id)
    store = session_store(session_id)

    client = chat_client()
    async with AsyncExitStack() as stack:
        # Connect the agent's MCP servers for the life of the request, then build the tool set the
        # model sees: local function tools + long-term-memory tools + MCP tools.
        mcp = await mcp_toolset(stack, build_mcp_urls())
        local = all_tools()
        tools = [*(spec for _, spec in local.values()), *memory_tools(), *mcp.schemas]

        # Load prior turns, append this turn's input, and persist the new user messages.
        history = await store.get_messages()
        new_messages: list[dict[str, Any]] = list(request.get("input") or [])
        await store.add_messages(new_messages)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, *new_messages]

        for _ in range(MAX_STEPS):
            # One model call, streamed. The assembled assistant message lands in `holder`.
            holder: dict[str, Any] = {}
            async for event in _run_completion(client, messages, tools, holder):
                yield event
            assistant = holder["message"]
            messages.append(assistant)
            await store.add_messages([assistant])

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                return  # the model answered; the loop is done

            # Run every requested tool, feed results back, and loop so the model can use them.
            for call in tool_calls:
                result = await _run_tool(call, local, mcp)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": result,
                }
                yield {"type": "message", "message": tool_message}
                messages.append(tool_message)
                await store.add_messages([tool_message])


async def _run_completion(
    client, messages: list[dict], tools: list[dict], holder: dict
) -> AsyncGenerator[dict, None]:
    """Stream one model call: yield token deltas, then the completed assistant message.

    Puts the assembled assistant message into ``holder['message']`` so the caller can inspect its
    tool calls after the stream drains.
    """
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools or None,
        stream=True,
    )
    assembler = _AssistantAssembler()
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield {"type": "delta", "content": delta.content, "id": chunk.id}
        assembler.add(delta)
    message = assembler.finish()
    holder["message"] = message
    yield {"type": "message", "message": message}


async def _run_tool(call: dict, local: dict, mcp) -> str:
    """Execute one tool call — local function, long-term memory, or MCP — and return its text result."""
    name = call["function"]["name"]
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}."
    try:
        if name in local:
            func, _ = local[name]
            return str(func(**arguments))
        if (memory_result := run_memory_tool(name, arguments)) is not None:
            return memory_result
        if mcp.handles(name):
            return await mcp.call(name, arguments)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"Error running {name}: {exc}"
    return f"Error: unknown tool {name!r}."


class _AssistantAssembler:
    """Accumulate a streamed Chat Completions assistant message from its delta chunks."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def add(self, delta: Any) -> None:
        if delta.content:
            self._content.append(delta.content)
        for tool_call in delta.tool_calls or []:
            call = self._tool_calls.setdefault(
                tool_call.index,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tool_call.id:
                call["id"] = tool_call.id
            if tool_call.function and tool_call.function.name:
                call["function"]["name"] = tool_call.function.name
            if tool_call.function and tool_call.function.arguments:
                call["function"]["arguments"] += tool_call.function.arguments

    def finish(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self._content)}
        if self._tool_calls:
            message["tool_calls"] = [self._tool_calls[i] for i in sorted(self._tool_calls)]
        return message
