import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "gemini-embedding-001"

if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY inside your .env file!")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def build_index(jsonl_path="byd_f3_faults_en.jsonl"):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    texts = [r["text"] for r in records]
    print(f"1. Loaded {len(records)} records from {jsonl_path}.")
    print(f"2. Generating multilingual embeddings via {EMBED_MODEL}...")
    
    res = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )
    matrix = np.array([e.values for e in res.embeddings], dtype=np.float32)
    
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norms
    
    np.save("vectors.npy", normalized_matrix)
    with open("records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    print("3. Done! Created 'vectors.npy' and 'records.json' successfully.")

if __name__ == "__main__":
    build_index()