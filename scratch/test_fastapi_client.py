from fastapi.testclient import TestClient
from legal_rag.api.main import create_app

app = create_app()

print("Testing QueryRequest schema validation with TestClient...")
with TestClient(app) as client:
    res = client.post("/rag/query/stream", json={
        "query": "What is Section 73 of the Indian Contract Act?",
        "model_mode": "fast",
        "conversation_context": []
    })
    print("TestClient Status Code:", res.status_code)
    if res.status_code != 200:
        print("Response text:", res.text)
