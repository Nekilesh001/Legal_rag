import requests
import json

url = "http://localhost:8000/rag/query/stream"
payload = {
    "query": "What does Section 73 of the Indian Contract Act say?",
    "model_mode": "fast",
    "conversation_context": []
}

print("=== TESTING POST /rag/query/stream API ENDPOINT ===")
try:
    response = requests.post(url, json=payload, stream=True, timeout=120)
    print("HTTP Status Code:", response.status_code)
    if response.status_code == 200:
        print("Streaming Events:")
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and len(decoded) < 200:
                    print(" ", decoded)
                elif decoded.startswith("event: "):
                    print(decoded)
        print("\nSTREAM TEST COMPLETED SUCCESSFULLY!")
    else:
        print("Error Response:", response.text)
except Exception as e:
    print("API Stream Test Exception:", e)
