"""Conversation session store for the hand-rolled agent.

No framework session object — the store is a plain ordered list of Chat Completions message dicts
(``{"role": ..., "content": ..., "tool_calls": ...}``). The agent loads prior turns with
``get_messages`` before calling the model and appends the turn's new messages with ``add_messages``.
``session_store(session_id)`` returns the store for a conversation.

Default (no config): an in-process ``_LocalSessionStore`` — a dict of lists kept in memory, cached
per session id. Multi-turn history is preserved within a single running process, no database. It does
NOT survive restarts or span replicas.

Durable (``AGENT_SESSION_STORE`` set): a ``DatabricksSessionStore``. Instead of a database the app
connects to directly, it stores each message as one **session item** through the managed Session
Store REST API (over RPCs only — no Lakebase/Postgres connection), so the transcript is durable
across restarts and replicas. Setting the env var is the only change; the agent code is identical.

The durable store uses a vendored REST client (``session_store_client.py``) so the template needs no
unpublished dependency; swap it for the published package when it lands.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Optional, Protocol, runtime_checkable

from databricks_mason.runtime.session_store_client import Session as _StoreSession
from databricks_mason.runtime.session_store_client import SessionStoreClient

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"

# Fetch items oldest-first so the transcript replays in write order.
_ORDER_BY = "create_time asc"

Message = dict[str, Any]


@runtime_checkable
class SessionStore(Protocol):
    """A conversation's message list. Two calls: read the transcript, append new messages."""

    async def get_messages(self) -> list[Message]: ...

    async def add_messages(self, messages: list[Message]) -> None: ...


# Process-wide message lists, one per session id — this is what makes multi-turn work in-process.
_local_messages: dict[str, list[Message]] = defaultdict(list)


def _session_actor(session_id: str) -> str:
    """Actor partition for the durable store. Defaults to the session id (one actor per session)."""
    return os.getenv(_SESSION_ACTOR_ENV) or session_id


def session_store(session_id: str) -> SessionStore:
    """The message store the agent persists conversation state to for ``session_id``.

    In-process ``_LocalSessionStore`` by default; a durable ``DatabricksSessionStore`` when
    ``AGENT_SESSION_STORE`` names a managed Session Store.
    """
    store = os.getenv(_SESSION_STORE_ENV)
    if store:
        return DatabricksSessionStore(session_id, store, actor_id=_session_actor(session_id))
    return _LocalSessionStore(session_id)


class _LocalSessionStore:
    """In-process, non-durable message list shared per session id (process-wide dict)."""

    def __init__(self, session_id: str) -> None:
        self._messages = _local_messages[session_id]

    async def get_messages(self) -> list[Message]:
        return list(self._messages)

    async def add_messages(self, messages: list[Message]) -> None:
        self._messages.extend(messages)


class DatabricksSessionStore:
    """A message store over the Databricks Session Store REST API.

    Each Chat Completions message is stored as one Session Store item whose ``data`` is the message
    dict (opaque JSON to the store). ``get_messages`` replays them in write order. The session is
    created on first use (get-or-create), keyed by the conversation's ``session_id`` under the
    configured actor.
    """

    def __init__(
        self,
        session_id: str,
        session_store_name: str,
        *,
        actor_id: str,
        client: Optional[SessionStoreClient] = None,
        workspace_client: Optional[Any] = None,
    ) -> None:
        if not session_store_name:
            raise ValueError("session_store_name is required")
        self.session_id = session_id
        self._actor_id = actor_id
        self._client = (client or SessionStoreClient(workspace_client)).set_session_store(
            session_store_name
        )
        self._session: _StoreSession | None = None

    async def get_messages(self) -> list[Message]:
        return await _run_sync(self._read_messages)

    async def add_messages(self, messages: list[Message]) -> None:
        if not messages:
            return
        await _run_sync(lambda: self._client.append_items(self._resolve(), items=list(messages)))

    # ----- internals ----------------------------------------------------------

    def _resolve(self) -> _StoreSession:
        if self._session is not None:
            return self._session
        try:
            self._session = self._client.get_session(session_id=self.session_id)
        except tuple(_not_found_errors()):  # type: ignore[misc]
            self._session = self._client.create_session(
                actor_id=self._actor_id,
                session_id=self.session_id,
                metadata={"client": "mason-vanilla-agent"},
            )
        return self._session

    def _read_messages(self) -> list[Message]:
        return [item.data for item in self._client.list_items(self._resolve(), order_by=_ORDER_BY)]


def _not_found_errors() -> tuple[type, ...]:
    """Exception types that mean 'session does not exist yet'."""
    try:
        from databricks.sdk.errors import NotFound

        return (NotFound,)
    except ImportError:  # pragma: no cover - SDK always present in practice
        return (_SessionNotFound,)


class _SessionNotFound(Exception):
    """Fallback 'not found' used only when databricks.sdk is unavailable."""


async def _run_sync(fn):
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, fn)
