import os
import requests
from dotenv import load_dotenv

load_dotenv()

JIRA_BASE_URL = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]


def create_jira_bug(summary: str, description: str) -> dict:
    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},   # which project the ticket goes into
            "issuetype": {"name": "Bug"},           # must match an issue type that exists in YOUR project
            "summary": summary, 
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": description}]}
                ],
            },
        }
    }

    response = requests.post(
        f"{JIRA_BASE_URL}/rest/api/3/issue",
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),  # requests builds the Basic auth header for you
        json=payload,
        timeout=10,
    )

    if response.status_code != 201:
        raise RuntimeError(f"Jira error {response.status_code}: {response.text}")

    key = response.json()["key"]  # e.g. "OPS-42"
    return {"key": key, "url": f"{JIRA_BASE_URL}/browse/{key}"}


if __name__ == "__main__":
    ticket = create_jira_bug(
        summary="Test bug from Python script",
        description="Created while learning REST APIs. Safe to delete.",
    )
    print(ticket)