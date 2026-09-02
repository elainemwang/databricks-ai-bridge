"""Extra MCP servers to offer the agent — this is where you configure them.

Empty by default: the agent connects only the MCP servers declared in ``agent.toml``. Add full MCP
endpoint URLs here to offer more; ``agent.py`` passes them to ``mcp_toolset``, which opens a session
to each (with Databricks auth), lists its tools, and makes them callable in the loop.
"""


def build_mcp_urls() -> list[str]:
    """Return extra MCP server URLs to offer the agent. Empty by default — add your own.

    Example (a Databricks-managed MCP, authed as the app service principal)::

        from databricks.sdk import WorkspaceClient

        host = WorkspaceClient().config.host.rstrip("/")
        return [f"{host}/api/2.0/mcp/functions/system/ai"]
    """
    return []
