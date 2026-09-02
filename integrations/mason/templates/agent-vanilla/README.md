# Agent — Vanilla (hand-rolled, FastAPI)

A **hand-rolled** agent for Databricks Apps — **no agent framework**. The reasoning loop is plain
Python over the **Chat Completions** API (call the model, run any tools it asks for, feed the results
back, repeat), served from a **FastAPI app**. It runs locally with **no database and no setup** —
just an auth profile.

This template is for understanding exactly what an agent is: read `agent/agent.py` top to bottom and
there is no magic — a model call, a tool loop, message dicts. Everything Databricks-shaped (the model
client, sessions, MCP, memory, tracing) comes from `databricks-mason` as small, independent pieces
you wire in.

Messages are Chat Completions dicts (`{"role", "content", "tool_calls"}`) everywhere — the session
store, the model call, and the browser UI all speak that one shape. `POST /invocations` takes an
`input` list of those message dicts (streaming via SSE, plus an in-memory `background` mode with
`GET /invocations/{id}`) and returns messages.

Local clients can use `/invocations`. For a deployed Databricks App, use the equivalent
`/api/invocations` route with an OAuth Bearer token; Databricks Apps reserves `/api/*` for
programmatic token authentication. Polling and health checks likewise have `/api` aliases.

## Project layout

```
agent/                 # the agent (reasoning plane) — this is what you edit
  agent.py             #   the hand-rolled loop: invoke/stream handlers + Chat Completions tool loop
  tools/               #   function tools — drop a *.py file here to add one (auto-collected)
    sample_tool.py     #     get_current_time — a working example (@tool)
    send_message.py    #     a second example tool with arguments
  mcps.py              #   extra MCP server URLs: none by default; add to build_mcp_urls() to offer some
runtime/               # the HTTP surface — SDK-agnostic; rarely edited
  runtime.py           #   build_app(): FastAPI routes, SSE framing, tracing spans, background wiring
  main.py              #   entry point: loads config, builds the app, runs uvicorn
tests/
  test_agent.py        #   hermetic smoke tests + one live model call
```

You edit `agent/agent.py`, `agent/tools/`, and `agent/mcps.py`; the plumbing (raw model client,
session store, MCP tool loading, long-term memory, tracing, background store) lives in the
`databricks-mason` package — framework-neutral pieces under `databricks_mason.runtime`, the
hand-rolled adapter under `databricks_mason.vanilla`. `runtime/runtime.py` is the SDK-agnostic HTTP
surface — it wires two generic handlers (`invoke_handler`/`stream_handler`) to the endpoints, so the
agent lives entirely behind them in `agent/agent.py`. `tools/` is a drop-in package: add a `*.py`
with a `@tool` function and it's auto-collected (no edits to existing code).

## The loop, in one paragraph

`stream_handler` builds the tool list the model sees — your local `@tool` functions, plus long-term
memory tools (if configured), plus any MCP tools — loads prior turns from the session store, then
runs up to `MAX_STEPS` passes: call the model (streaming its text as `delta` events), and if the
assistant message contains `tool_calls`, run each tool and append its result as a `tool` message, then
loop so the model can use the results. When the model replies without tool calls, the turn is done.
Every message is persisted to the session store as it's produced.

## Run locally

No database required. Conversation state is kept in an in-process message list.

```bash
# 1. Configure a Databricks auth profile (used only to call the model)
cp .env.example .env
# edit .env: set DATABRICKS_CONFIG_PROFILE=<your-profile>

# 2. Start the server (installs deps via uv on first run)
uv run start-server        # serves at http://localhost:8000

# 3. Send a request
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What time is it? Use your tool."}]}'
```

The model call goes to your Databricks workspace (via the profile), through a raw `openai` client
pointed at the workspace's `/serving-endpoints` gateway. Everything else — session storage, tracing —
is off by default and requires no setup.

## Client contract

The Databricks Apps `__Host-databricks-app-router` cookie is the single session identifier. It both
keeps requests on the same App replica and keys the session store. Do **not** send `session_id` in
request bodies — the runtime ignores an old body value and injects the cookie value before calling
the agent. Browsers resend the Apps cookie automatically; API clients must preserve it in a cookie
jar. Localhost has no Apps router, so the server sets an HTTP-only `mason-local-session` fallback
cookie instead.

TODO: switch the client contract to `X-Routing-Key` when Databricks Apps supports it. Until then use
the [documented Apps routing cookie](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/horizontal-scaling#api-clients).

```bash
BASE=http://localhost:8000
COOKIE_JAR=/tmp/mason-agent.cookies
curl -s -c "$COOKIE_JAR" "$BASE/health"
```

**Non-streaming:**

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"hi"}]}'
```

The response is `{ "output": [...], "session_id": "...", "status": "completed" }`. `output` contains
the run's new Chat Completions message dicts (assistant turns, tool calls, tool results).

**Streaming** adds `"stream": true` and returns SSE. Completed messages use
`data: {"type":"message","message":{...}}`; token chunks use
`data: {"type":"delta","content":"...","id":"..."}`; the final frame is `data: [DONE]`.

```bash
curl -sN -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"Count to three"}],"stream":true}'
```

**Background** (add `"background": true`) returns an `inv_...` id immediately; poll it:

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"do something"}],"background":true}'
# -> {"id":"inv_...","status":"in_progress"}

curl -s -b "$COOKIE_JAR" "$BASE/invocations/inv_..."
# -> in_progress, completed + output, or failed + error
```

Background runs and polling are in-memory and single-process. The routing cookie is required so the
poll reaches the same replica, but the run itself does not survive a restart.

**Multi-turn** needs no body bookkeeping: reuse the same cookie jar for each turn.

## Customize the agent

- **Model / instructions:** `MODEL` and `SYSTEM_PROMPT` in `agent/agent.py`.
- **Add a tool:** drop a new file in `agent/tools/` with a `@tool`-decorated function; it's collected
  automatically (see `agent/tools/sample_tool.py`). The schema is derived from the signature.
- **Add an MCP server:** append its endpoint URL to `build_mcp_urls()` in `agent/mcps.py`, or declare
  it in `agent.toml` (both are connected each request).
- **Make state durable:** set `AGENT_SESSION_STORE` (see "Enable durable state" below); the store
  lives in `databricks_mason/vanilla/sessions.py`.
- **Add long-term memory:** set `AGENT_MEMORY_STORE` to a managed memory store ID; the loop then
  offers the `remember`/`recall` tools from `databricks_mason/vanilla/memory.py`.
- **Change the loop:** it's all in `agent/agent.py` — the step cap, how tool calls are dispatched, how
  results are fed back. No framework to fight.
- **Change the HTTP surface:** `runtime/runtime.py` — routes, SSE framing, background wiring.

## Test

```bash
uv run pytest                 # hermetic smoke tests (tools, assembler, session store)
```

The smoke tests need no auth. `tests/test_agent.py` also has one end-to-end test that runs the full
loop against the model; it runs only when a workspace profile is configured
(`DATABRICKS_CONFIG_PROFILE` or `DATABRICKS_HOST`+`DATABRICKS_TOKEN`) and skips otherwise.

## Deploy

Deploy with the [Mason](../../README.md) CLI:

```bash
mason deploy agent-vanilla --source .
```

Add `--with-memory-store <name> --with-session-store <name> --actor-id <actor>` to wire managed
state. Mason provisions or resolves the stores, injects the store/actor env vars, and deploys the
App.

`app.yaml` carries the app's start command and env. By default the deployed app is the same lean
backend: in-process session state, tracing off.

### Enable MLflow tracing (optional)

Tracing turns on when MLflow has **both a destination and an experiment** — set one of each, in
whichever form you have. The app code needs no change; MLflow resolves the specific value.

- **Destination:** `MLFLOW_TRACKING_URI` (e.g. `"databricks"`) or `MLFLOW_TRACING_DESTINATION`.
- **Experiment:** `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`.

The hand-rolled loop calls Chat Completions through a raw OpenAI client, so tracing uses MLflow's
OpenAI autolog (`mlflow.openai.autolog()`), plus a per-request span from `runtime/runtime.py` tagged
with the session id.

### Enable durable state (optional)

By default the agent keeps conversation history in an in-process message list — multi-turn works
within a running process but does not survive restarts or span replicas.

Set **`AGENT_SESSION_STORE`** to a managed [Session Store](../../README.md) name and
`databricks_mason/vanilla/sessions.py` stores each message through the managed Session Store **REST
API** instead — durable conversation history across restarts and replicas, over RPCs (no database the
app connects to directly). No agent code changes; the store swap is the only difference. The store
must already exist; access uses the caller's normal Databricks auth (the deployed app's service
principal, or your profile locally — whichever the Session Store grants).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABRICKS_CONFIG_PROFILE` | `DEFAULT` | Auth profile used to call the model (local dev) |
| `PORT` | `8000` | Port the server listens on |
| `AGENT_MEMORY_STORE` | _unset_ | Managed memory store ID → offers `remember`/`recall` long-term-memory tools |
| `AGENT_MEMORY_ACTOR_ID` | `agent` | Actor partition used by memory tools |
| `AGENT_SESSION_STORE` | _unset_ | Managed Session Store name → durable transcript (REST-backed); unset = in-process message list |
| `AGENT_SESSION_ACTOR_ID` | session id | Actor used by the durable store |
| `MLFLOW_TRACKING_URI` | _unset_ | Trace destination (e.g. `databricks`). A destination + an experiment enables tracing |
| `MLFLOW_TRACING_DESTINATION` | _unset_ | Alt destination (either destination var works) |
| `MLFLOW_EXPERIMENT_ID` | _unset_ | Experiment to trace to (by id) |
| `MLFLOW_EXPERIMENT_NAME` | _unset_ | Experiment to trace to (by name; alternative to the id) |

## Notes

- **The loop in `agent/agent.py` is the point** — it's an ordinary Chat Completions tool loop, no
  framework. `runtime/runtime.py` is SDK-agnostic: it hosts any agent exposing the
  `invoke_handler`/`stream_handler` dict contract.
- **Background mode is in-memory** (`databricks_mason/runtime/background.py`, wired in
  `runtime/runtime.py`) — non-durable, single-process; see the note under the client contract.
- **No human-in-the-loop.** Unlike the LangGraph and OpenAI templates, this one does not gate tools on
  approval. Add a pause/resume around the tool dispatch in `agent.py` if an action needs it.
