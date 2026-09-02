"""A raw OpenAI client pointed at Databricks model serving — no framework, no wrapper.

``chat_client()`` returns a plain ``openai.AsyncOpenAI`` whose ``base_url`` is the workspace's
``/serving-endpoints`` gateway and whose requests carry Databricks auth. The agent calls
``client.chat.completions.create(model=..., messages=..., tools=...)`` exactly as it would against
OpenAI; only the transport is Databricks. Databricks model endpoints reject the ``strict`` field on
tool schemas for non-GPT models, so the agent should omit it (the template does).

Auth rides an httpx client, not an ``api_key``: ``config.authenticate()`` returns the Authorization
header for whatever profile is configured (PAT, OAuth, run-local), refreshed per request. This is the
same mechanism ``databricks-openai`` uses; it is inlined here so the vanilla template shows the whole
wiring with no extra package.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

import httpx
from openai import AsyncOpenAI

from databricks_mason.runtime.workspace import workspace_client


class _DatabricksBearerAuth(httpx.Auth):
    """Attach the SDK's current Authorization header to each request (refreshes per call)."""

    def __init__(self, authenticate) -> None:
        self._authenticate = authenticate

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = self._authenticate()["Authorization"]
        yield request


@lru_cache(maxsize=1)
def chat_client() -> AsyncOpenAI:
    """A raw ``AsyncOpenAI`` client routed at the workspace's model-serving gateway.

    Cached so the underlying HTTP connection pool is reused across requests. ``api_key`` is a
    placeholder — real auth is the Databricks Authorization header injected by the httpx client.
    """
    config = workspace_client().config
    base_url = f"{config.host.rstrip('/')}/serving-endpoints"
    http_client = httpx.AsyncClient(auth=_DatabricksBearerAuth(config.authenticate))
    return AsyncOpenAI(base_url=base_url, api_key="no-token", http_client=http_client)
