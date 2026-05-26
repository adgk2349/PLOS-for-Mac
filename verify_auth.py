import os

import requests


url = os.getenv("VERIFY_AUTH_URL", "http://localhost:8001/v1/chat/completions")
token = os.getenv("VERIFY_AUTH_TOKEN", "replace-me")
model = os.getenv("VERIFY_AUTH_MODEL", "gemma-4-e4b")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10,
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
