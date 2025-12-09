#!/usr/bin/env python3
"""
Test script to reproduce 422 error
"""
import requests
import json

# Test 1: Valid request (what Django should be sending)
valid_payload = {
    "question": "테스트 질문",
    "user_id": "test_user_123",
    "chat_history": [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"}
    ],
    "session_id": "session_123"
}

# Test 2: Invalid request (missing user_id)
invalid_payload_no_user = {
    "question": "테스트 질문",
    "chat_history": [],
    "session_id": "session_123"
}

# Test 3: Invalid request (empty question)
invalid_payload_empty_question = {
    "question": "",
    "user_id": "test_user_123",
    "chat_history": [],
    "session_id": "session_123"
}

# Test 4: Invalid request (wrong chat_history format)
invalid_payload_bad_history = {
    "question": "테스트 질문",
    "user_id": "test_user_123",
    "chat_history": ["wrong format"],
    "session_id": "session_123"
}

SERVER_URL = "http://localhost:8080"

def test_request(name, payload):
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(
            f"{SERVER_URL}/api/v1/chat",
            json=payload,
            timeout=5
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 422:
            print("❌ 422 Unprocessable Entity")
            print(f"Error Details: {response.json()}")
        elif response.status_code == 200:
            print("✅ Success")
            result = response.json()
            print(f"Answer length: {len(result.get('answer', ''))}")
        else:
            print(f"Other status: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python serve_unified.py --rag-type langgraph")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing API Request Formats")
    print("Make sure serve_unified.py is running on localhost:8080")

    test_request("Valid Request (Django format)", valid_payload)
    test_request("Invalid: Missing user_id", invalid_payload_no_user)
    test_request("Invalid: Empty question", invalid_payload_empty_question)
    test_request("Invalid: Bad chat_history format", invalid_payload_bad_history)
