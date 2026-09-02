# Agent Development Guide

A hand-rolled agent backend for Databricks Apps — no agent framework. The reasoning loop is plain
Python over the Chat Completions API, served from a FastAPI app. Local-first: runs with no database
and no setup beyond a Databricks auth profile. MLflow tracing is optional.

See `README.md` for the full run / deploy / client-contract docs. This file is the quick map for
making changes.

## Run it

```bash
cp .env.example .env          # set DATABRICKS_CONFIG_PROFILE=<your-profile>
uv run start-server           # http://localhost:8000
```

No database needed — conversation state uses an in-process message list by default.

## Sample requests

`input` is a list of Chat Completions message dicts; the reply is `{ "output": [...], "session_id": "..." }`
where `output` is the run's new messages (same shape). The `__Host-databricks-app-router` cookie is
both the Apps routing key and application session id; never send `session_id` in the JSON body. Use a
cookie jar locally so the server's `mason-local-session` fallback is reused.

```bash
COOKIE_JAR=/tmp/mason-agent.cookies
curl -s -c "$COOKIE_JAR" http://localhost:8000/health

# Sync — run a turn to completion
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What time is it? Use your tool."}]}'

# Streaming — SSE frames ending with `data: [DONE]`
curl -sNb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Count to 3."}], "stream": true}'
# frames: {"type":"message","message":{...}} (completed) and {"type":"delta","content":"...","id":"..."}

# Multi-turn — reuse the same cookie jar (same process; in-memory store)
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "My name is Alice."}]}'
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What is my name?"}]}'
```

## Where things live

| You want to… | Edit |
| --- | --- |
| Change model / instructions | `agent/agent.py` (`MODEL`, `SYSTEM_PROMPT`) |
| Change the tool loop (step cap, dispatch, feedback) | `agent/agent.py` (`stream_handler`, `_run_tool`) |
| Add a function tool | new `*.py` in `agent/tools/` with a `@tool` function (auto-collected) |
| Add an MCP server | append its URL to `build_mcp_urls()` in `agent/mcps.py`, or declare it in `agent.toml` |
| Change how a request maps to a run | `agent/agent.py` (`invoke_handler` / `stream_handler`) |
| Change the session store | `databricks_mason/vanilla/sessions.py` |
| Change the model client | `databricks_mason/vanilla/model.py` |
| Change the HTTP surface (routes, SSE, background wiring) | `runtime/runtime.py` |
| Add a test | `tests/` (hermetic; gate model calls on a workspace profile — see `test_agent.py`) |

`runtime/runtime.py` is **SDK-agnostic** — it wires two generic handlers (`invoke_handler`/`stream_handler`,
plain `dict -> dict` / `dict -> AsyncGenerator[dict]`) to the endpoints. The agent lives entirely
behind those handlers in `agent/agent.py`, so the serving layer is the same regardless of framework.

`databricks_mason.runtime` (from the `databricks-mason` package) holds framework-neutral plumbing
(tracing, workspace client, background store, the Session Store REST client); `databricks_mason.vanilla`
holds the hand-rolled adapter (raw model client, session store, MCP tools, memory tools).

## How tools register

`agent/tools/all_tools()` auto-imports every module in the package and collects every `@tool`-decorated
function into `{name: (func, schema)}`. The `@tool` decorator (defined in `agent/tools/__init__.py`)
derives the Chat Completions schema from the function's signature and docstring. So a tool registers
just by existing in a file there — `stream_handler()` reads `all_tools()`. **Do not** edit `agent/agent.py`
to add a tool — just add a file to `agent/tools/`.

## The loop

`stream_handler` runs the classic tool-calling loop by hand:

1. Build the tool list: local `@tool` functions + long-term-memory tools (if `AGENT_MEMORY_STORE`) +
   MCP tools (from `agent.toml` and `build_mcp_urls()`).
2. Load prior turns from the session store; append this turn's input.
3. Up to `MAX_STEPS` times: call the model (streaming text as `delta` events), append the assistant
   message. If it has `tool_calls`, run each (`_run_tool` dispatches to local / memory / MCP), append
   each result as a `tool` message, and loop. If not, the turn is done.
4. Every message is persisted to the session store as produced.

## Sessions

- Default: `databricks_mason/vanilla/sessions.py`'s `session_store()` returns an in-process message
  list keyed per session id — no database, multi-turn works in-process.
- The `__Host-databricks-app-router` cookie is both the Apps routing key and the session id. The
  runtime injects it into the internal handler request; body `session_id` values are ignored.
- For durable state, set `AGENT_SESSION_STORE` to a managed Session Store name: the store persists
  each message via the REST API (no DB connection), so the transcript survives restarts/replicas.
  `AGENT_SESSION_ACTOR_ID` selects the actor partition.
- Background mode is in-memory / single-process — non-durable. The store is
  `databricks_mason/runtime/background.py` (wired in `runtime/runtime.py`).
- No human-in-the-loop: this template does not gate tools on approval. Add a pause/resume around the
  tool dispatch in `agent.py` if you need it.

## MLflow tracing

Optional. Set both a destination (`MLFLOW_TRACKING_URI` or `MLFLOW_TRACING_DESTINATION`) and an
experiment (`MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`) to enable (`mlflow.openai.autolog()`,
since the loop calls Chat Completions through a raw OpenAI client); leave either half unset to skip.
`runtime/runtime.py` opens a per-request span regardless.

## Quick commands

| Task | Command |
| --- | --- |
| Run locally | `uv run start-server` |
| Test | `uv run pytest` (hermetic; live model test runs only with a profile) |
| Deploy | `mason deploy agent-vanilla --source .` |
