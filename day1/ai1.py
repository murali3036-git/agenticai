

import os
from openai import OpenAI

# NOTE: Replace "YOUR_API_KEY" with your actual, valid Gemini API key.
API_KEY = ""
MODEL_NAME = "gemini-2.5-pro"

try:
    client = OpenAI(
        api_key=API_KEY, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say a joke on computer less than 100 characters"}
        ]
        # Keeping max_tokens high (e.g., 500) to confirm it is NOT the issue.
    )

    # If the code reaches here, the call succeeded.
    print("\n--- Success! Joke Content ---")
    print(response.choices[0])
    print(response.choices[0].message.content)
    print(f"Finish Reason: {response.choices[0].finish_reason}")

except Exception as e:
    # This will catch and print connection/authentication errors.
    print(f"\n❌ An API Error Occurred:")
    print(e)
    print("\nAction Required: Check your API Key and Base URL carefully.")
