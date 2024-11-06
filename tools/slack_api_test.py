from kubiya_sdk.tools import function_tool


@function_tool(
    description="sends a message to a slack channel",
    requirements=["slack-sdk==3.33.3"],
    secrets=["SLACK_API_TOKEN"],
)
def slack_api_test(
    channel_id: str,
    message: str,
):
    import os
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    # Initialize the client with your bot token
    token = os.environ["SLACK_API_TOKEN"]
    client = WebClient(token=token)

    # Send a message
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=message,
        )
        print("Message sent successfully")
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")
