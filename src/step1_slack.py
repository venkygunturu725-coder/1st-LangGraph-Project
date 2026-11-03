import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]


def post_to_slack(text: str) -> dict:
    response = requests.post(
        "https://slack.com/api/chat.postMessage",  
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"channel": SLACK_CHANNEL_ID, "text": text},   
        timeout=(5, 10)
    )

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Slack error: {data.get('error')}")

    return data

def read_slack_messages() -> list:
    response = requests.get(
        "https://slack.com/api/conversations.history",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        params={"channel": SLACK_CHANNEL_ID, "limit": 10},
        timeout=(5, 10)
    )

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Slack error: {data.get('error')}")

    return data

if __name__ == "__main__":
    result = post_to_slack("Hello venky")
    print("Posted! message timestamp:", result["ts"])


