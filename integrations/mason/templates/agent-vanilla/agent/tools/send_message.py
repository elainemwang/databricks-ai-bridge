"""A second sample tool with arguments — shows a multi-parameter tool the model must fill in.

Unlike ``get_current_time`` (no args), the model has to extract ``recipient`` and ``body`` from the
conversation to call this. Swap the body for a real send (email/Slack/SMS). This template does not
gate tools on human approval; add that in your loop if an action needs it.
"""

from agent.tools import tool


@tool
def send_message(recipient: str, body: str) -> str:
    """Send a message to a recipient. Use when the user asks to notify or message someone."""
    # A real implementation would call an email/Slack/SMS API here.
    return f"Message sent to {recipient}: {body}"
