import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv("CLARIN_API_TOKEN")

def get_questions(text):
    request_body = {
        "model": "llama3.3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that creates thoughtful and neutral questions about input texts. "
                    "You classify them into three types: comprehension, analytical, and stylistic."
                )
            },
            {
                "role": "user",
                "content": f"""Here is a text passage:
                {text}

                Based on this text, generate three questions, one for each of the following categories:

                1. Comprehension - A question that checks basic understanding of the main idea or key content.
                2. Analytical - A question that requires interpretation, critical thinking, or understanding of reasoning within the text.
                3. Stylistic - A question about structure, tone, or writing style.

                **Do not update id field**
                Please format your response as a JSON object (without any string like ```json ... ```), like this:

                {{
                    "comprehension_question": "...",
                    "comprehension_answer": "...",
                    "analytical_question": "...",
                    "analytical_answer": "...",
                    "textual_stylistic_question": "...",
                    "textual_stylistic_answer": "..."
                }}

                No explanations, no formatting — only raw JSON."""
            }
        ]
    }

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'api-token': api_token
    }

    response = requests.post(
        'https://services.clarin-pl.eu/api/v1/oapi/chat/completions',
        headers=headers,
        json=request_body
    )

    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Failed to parse content: {e}")
            return None
    else:
        print(f"Request failed ({response.status_code}): {response.text}")
        return None