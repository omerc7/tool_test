from json import tool
from kubiya_sdk import Tool, tool_registry
from kubiya_sdk.tools import function_tool


@function_tool(
    description="answers questions based on the slack channel history",
    requirements=["litellm==1.71.1", "requests==2.32.3", "pydantic==2.11.5"],
    image="python:3.12-alpine",
    env=[
        "LLM_BASE_URL",
        "SLACK_DOMAIN",
        "KUBIYA_API_URL",
        "KUBIYA_USER_ORG",
        "SLACK_CHANNEL_ID",
        "KUBIYA_USER_EMAIL",
        "KUBIYA_USER_MESSAGE",
    ],
    secrets=["LLM_API_KEY", "KUBIYA_API_KEY"],
)
def slack_knowledge():
    import os
    import json
    import litellm
    import requests
    from pydantic import BaseModel

    class Question(BaseModel):
        question: str

    class AnswerReference(BaseModel):
        ts: str
        channel_id: str

    class Answer(BaseModel):
        content: str
        references: list[AnswerReference]

    def format_slack_threads(messages):
        formatted = []
        for msg in messages:
            main_content = msg["content"].strip()
            metadata = msg.get("metadata", {})
            channel_id = metadata.get("channel_id", "N/A")
            ts = metadata.get("ts", "N/A")

            formatted_msg = (
                f"Main Message:\n"
                f"{main_content}\n"
                f"(channel_id: {channel_id}, ts: {ts})"
            )

            thread_replies = msg.get("thread", [])
            if thread_replies:
                formatted_msg += "\n\nThread Replies:"
                for reply in thread_replies:
                    reply_content = reply["content"].strip()
                    reply_metadata = reply.get("metadata", {})
                    reply_channel_id = reply_metadata.get("channel_id", "N/A")
                    reply_ts = reply_metadata.get("ts", "N/A")
                    formatted_msg += (
                        f"\n  - {reply_content} "
                        f"(Channel ID: {reply_channel_id}, Timestamp: {reply_ts})"
                    )

            formatted.append(formatted_msg)

        return "\n\n---\n\n".join(formatted)

    def query_rag(query: str):
        try:
            kubiya_api_url = os.environ["KUBIYA_API_URL"]
            payload = {
                "threshold": 0.5,
                "query": query,
                # "channel_id": os.environ["SLACK_CHANNEL_ID"], # TODO: add channel id
            }
            headers = {
                "Authorization": f"UserKey {os.environ['KUBIYA_API_KEY']}",
            }
            result = requests.post(f"{kubiya_api_url}/api/v1/rag/query/mind", json=payload, headers=headers)
            result.raise_for_status()

            return result.json()
        except Exception as e:
            print(e)
            print("tool ended with error")
            return

    def pretty_print_answer(answer: Answer):
        slack_domain = os.environ["SLACK_DOMAIN"]
        print(f"Answer: {answer.content}")
        # TODO: add references with slack links
        print("\n  References:")
        for ref in answer.references:
            print(
                f" - https://{slack_domain}.slack.com/archives/{ref.channel_id}/p{ref.ts}"
            )

    llm_key = os.environ["LLM_API_KEY"]
    llm_base_url = os.environ["LLM_BASE_URL"]

    try:
        response = litellm.completion(
            model="openai/gpt-4o",
            api_key=llm_key,
            base_url=llm_base_url,
            response_format=Question,
            messages=[
                {
                    "content": "Take the user's query and expand it into a more detailed and informative version to help retrieve better results. "
                    "Add assumptions, context, and related keywords if needed. Be concise but specific. Return the expanded query only! No other text.",
                    "role": "system",
                },
                {
                    "content": f"User message: {os.environ['KUBIYA_USER_MESSAGE']}",
                    "role": "user",
                },
            ],
        )

        expended_query = Question(**json.loads(response.choices[0].message.content)).question

        result = query_rag(
            expended_query
        )
        messages_with_thread_replies = [m for m in result if m.get("thread")]
        formated_result = format_slack_threads(messages_with_thread_replies)

        response = litellm.completion(
            model="openai/gpt-4o",
            api_key=llm_key,
            base_url=llm_base_url,
            response_format=Answer,
            messages=[
                {
                    "content": """
    You are a helpful assistant that can answer questions based on the provided context ONLY. You are given a query and a result from a knowledge base. You need to answer the query based on the result.
    Keep your response concise and to the point.
    """,
                    "role": "system",
                },
                {
                    "content": f"Query: {os.environ['KUBIYA_USER_MESSAGE']}\n\v Knowledge base result:\n{formated_result}",
                    "role": "user",
                },
            ],
        )
        answer = Answer(**json.loads(response.choices[0].message.content))

        pretty_print_answer(answer)

    except Exception as e:
        print(e)
        print("tool ended with error")
        return
