from langchain_core.tools import tool

from step1_slack import post_to_slack, read_slack_messages
from step2_jira import create_jira_bug


# Design rationale: the LLM never sees your Python code. It sees ONLY three things:
# the tool's name, its docstring, and its argument names/types. So the docstring
# is not documentation for humans, it is the instruction manual for the model.
# Be specific about WHEN to use the tool and WHAT each argument means.

@tool
def read_slack_messages_tool() -> str:
    """Read the last 10 messages from the team's Slack channel.

    Use this ONLY when the user explicitly asks to read or show Slack messages.
    Do NOT use this when reporting a failure.
    """
    try:
        data = read_slack_messages()
        messages = data.get("messages", [])

        lines = [
            f"{m.get('user') or m.get('bot_id') or 'unknown'}: {m.get('text', '')}"
            for m in reversed(messages)
        ]
        return "\n".join(lines) or "No messages found in the channel."
    except Exception as e:
        return f"ERROR reading Slack messages: {e}"
    
@tool
def create_jira_bug_tool(summary: str, description: str) -> str:
    """Create a Jira Bug ticket for a confirmed failure.

    Args:
        summary: Short one-line title saying what failed and where.
        description: Root cause, evidence, and suggested next steps.

    Returns the ticket key and URL.
    """
    try:
        ticket = create_jira_bug(summary, description)
        return f"Created Jira ticket {ticket['key']}: {ticket['url']}"
    except Exception as e:
        return f"ERROR creating Jira ticket: {e}"


@tool
def post_slack_message_tool(text: str) -> str:
    """Post a message to the team's Slack channel.

    Use this to share information with the team, for example to announce a newly
    created Jira ticket. Include the ticket key and URL in the text.

    Args:
        text: The message to post.
    """
    try:
        post_to_slack(text)
        return "Message posted to Slack successfully."
    except Exception as e:
        return f"ERROR posting to Slack: {e}"


tools = [read_slack_messages_tool, create_jira_bug_tool, post_slack_message_tool]