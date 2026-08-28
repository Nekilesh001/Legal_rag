import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

client = OpenAI(api_key=api_key, base_url=base_url)

models_to_test = [
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.1-70b-instruct",
    "openai/gpt-oss-120b",
    "meta/llama3-70b-instruct",
]

messages = [
    {"role": "system", "content": "You are a legal research assistant. Answer concisely."},
    {"role": "user", "content": "What is Section 73 of the Indian Contract Act?"}
]

print("=== BENCHMARKING NVIDIA API MODELS ===")
for model in models_to_test:
    print(f"\n--- Testing Model: {model} ---")
    t0 = time.perf_counter()
    ttft = None
    first_token_time = None
    content_acc = ""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
            stream=True
        )
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            txt = getattr(delta, "content", None)
            if txt:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                content_acc += txt
        t_total = (time.perf_counter() - t0) * 1000
        print(f"Status: SUCCESS")
        print(f"Time to First Token (TTFT): {ttft:.2f} ms" if ttft else "TTFT: N/A")
        print(f"Total Response Latency:      {t_total:.2f} ms ({t_total/1000:.2f} s)")
        print(f"Content Length:              {len(content_acc)} chars")
        print(f"Snippet: {content_acc[:100]}...")
    except Exception as e:
        print(f"Status: FAILED with error: {e}")
