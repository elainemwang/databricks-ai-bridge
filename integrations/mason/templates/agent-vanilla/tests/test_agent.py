"""Smoke tests for the hand-rolled agent.

Hermetic tests import only the leaf pieces (tool registration + schema derivation, the streaming
assistant assembler, the session store) — no Databricks auth needed, so they run anywhere. The live
test runs the full loop against the model; it is skipped unless a workspace profile is configured.
"""

import os

import pytest
from agent.agent import _AssistantAssembler, _session_id
from agent.tools import all_tools, tool
from databricks_mason.vanilla.sessions import _LocalSessionStore, session_store


def test_tools_autoregister_with_schemas():
    tools = all_tools()
    assert {"get_current_time", "send_message"} <= set(tools)
    func, schema = tools["send_message"]
    assert callable(func)
    # The schema is derived from the signature: both params, both required, string-typed.
    params = schema["function"]["parameters"]
    assert params["properties"] == {"recipient": {"type": "string"}, "body": {"type": "string"}}
    assert set(params["required"]) == {"recipient", "body"}
    assert schema["function"]["description"]  # docstring becomes the description


def test_tool_decorator_accepts_explicit_schema():
    explicit = {"type": "function", "function": {"name": "custom", "parameters": {}}}

    @tool(schema=explicit)
    def custom_tool():
        return "ok"

    assert getattr(custom_tool, "__mason_tool_schema__") is explicit


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _ToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


def test_assembler_collects_streamed_text():
    assembler = _AssistantAssembler()
    assembler.add(_Delta(content="Hel"))
    assembler.add(_Delta(content="lo"))
    assert assembler.finish() == {"role": "assistant", "content": "Hello"}


def test_assembler_reassembles_tool_calls_across_chunks():
    assembler = _AssistantAssembler()
    assembler.add(_Delta(tool_calls=[_ToolCallDelta(0, id="c1", name="send_message")]))
    assembler.add(_Delta(tool_calls=[_ToolCallDelta(0, arguments='{"recipient":"a",')]))
    assembler.add(_Delta(tool_calls=[_ToolCallDelta(0, arguments='"body":"hi"}')]))
    message = assembler.finish()
    assert message["content"] == ""
    assert message["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "send_message", "arguments": '{"recipient":"a","body":"hi"}'},
        }
    ]


def test_session_id_from_request():
    assert _session_id({"input": [], "session_id": "abc-123"}) == "abc-123"


def test_session_id_is_required_from_runtime():
    with pytest.raises(KeyError):
        _session_id({"input": []})


def test_local_session_store_selected_without_env(monkeypatch):
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    assert isinstance(session_store("s1"), _LocalSessionStore)


def test_durable_session_store_selected_with_env(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    # Stub the REST client so construction stays hermetic (no network).
    import databricks_mason.vanilla.sessions as sessions

    monkeypatch.setattr(sessions, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    store = session_store("s1")
    assert isinstance(store, sessions.DatabricksSessionStore)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


@pytest.mark.asyncio
async def test_local_session_store_round_trips_messages(monkeypatch):
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    import databricks_mason.vanilla.sessions as sessions

    sessions._local_messages.clear()
    store = session_store("round-trip")
    await store.add_messages([{"role": "user", "content": "hi"}])
    await store.add_messages([{"role": "assistant", "content": "hello"}])
    # A fresh store for the same id sees the shared, process-wide history.
    assert await session_store("round-trip").get_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def _has_workspace_auth() -> bool:
    return bool(
        os.getenv("DATABRICKS_CONFIG_PROFILE")
        or (os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    )


@pytest.mark.skipif(
    not _has_workspace_auth(),
    reason="no Databricks profile configured; skipping live model call",
)
@pytest.mark.asyncio
async def test_agent_responds_end_to_end():
    from agent.agent import configure, stream_handler

    configure()
    request = {"input": [{"role": "user", "content": "Reply with the single word: pong"}], "session_id": "test-e2e"}
    messages = [event["message"] async for event in stream_handler(request) if event["type"] == "message"]
    assert any(m.get("role") == "assistant" and m.get("content") for m in messages)
