import os
import json
import numpy as np
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

EMBED_MODEL = "gemini-embedding-001"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY) if GROQ_API_KEY else None

RECORDS = []
VECTORS = None
ACTIVE_GROQ_MODEL = None

def select_best_groq_model() -> str | None:
    """Queries Groq on startup to pick an active open-source chat model automatically."""
    if not groq_client:
        return None
    try:
        models_resp = groq_client.models.list()
        available = [
            m.id for m in models_resp.data
            if not any(skip in m.id.lower() for skip in ["whisper", "tts", "guard", "vision", "embed", "audio"])
        ]
        print(f"Available Groq chat models on your account: {available}")
        # Prefer strong multilingual open-source models
        preferred_keywords = ["llama-4", "qwen", "llama-3", "deepseek", "gemma", "mixtral"]
        for kw in preferred_keywords:
            for m_id in available:
                if kw in m_id.lower():
                    return m_id
        return available[0] if available else None
    except Exception as e:
        print(f"Could not list Groq models ({e}). Will use Gemini fallback.")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global RECORDS, VECTORS, ACTIVE_GROQ_MODEL
    with open("records.json", "r", encoding="utf-8") as f:
        RECORDS = json.load(f)
    VECTORS = np.load("vectors.npy")
    ACTIVE_GROQ_MODEL = select_best_groq_model()
    print(f"Server ready! Loaded {len(RECORDS)} records. Selected Groq LLM: {ACTIVE_GROQ_MODEL}")
    yield

app = FastAPI(title="Car Diagnostic RAG API", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "records_loaded": len(RECORDS),
        "vectors_ready": VECTORS is not None
    }

@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
def serve_demo():
    return FileResponse("demo.html")

SYSTEM_PROMPT = """You are a strict automotive diagnostic assistant.
Rules:
1. Answer the user's question ONLY using the facts inside [SOURCES].
2. Do NOT invent or add external mechanical information outside [SOURCES].
3. Reply in the SAME language the user used in their question (Arabic or English).
4. Clearly format the response into: Likely Fault, Possible Causes, Diagnosis Steps, Fix, and Can it be driven.
5. If the user's issue is not related to anything in [SOURCES], state that this issue is not in the database."""

class DiagnoseRequest(BaseModel):
    make: str = "BYD"
    model: str = "F3"
    question: str
    top_k: int = 2

class MatchedFault(BaseModel):
    id: str
    make: str
    model: str
    system: str
    title: str
    severity: str
    drive_advice: str
    obd_codes: list[str]
    score: float

class DiagnoseResponse(BaseModel):
    answer: str
    llm_used: str
    matched_faults: list[MatchedFault]

@app.post("/api/v1/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    valid_indices = [
        i for i, r in enumerate(RECORDS)
        if r.get("make", "").lower() == req.make.lower()
        and r.get("model", "").lower() == req.model.lower()
    ]
    if not valid_indices:
        raise HTTPException(status_code=404, detail=f"No records found for {req.make} {req.model}")

    # 1. Embed user question
    q_res = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=[req.question]
    )
    q_vec = np.array(q_res.embeddings[0].values, dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # 2. Cosine similarity + OBD2 code boost
    car_vectors = VECTORS[valid_indices]
    scores = np.dot(car_vectors, q_vec)

    for local_i, global_i in enumerate(valid_indices):
        for code in RECORDS[global_i].get("obd_codes", []):
            if code.lower() in req.question.lower():
                scores[local_i] += 0.5

    top_local = np.argsort(scores)[::-1][:req.top_k]
    matches = [(RECORDS[valid_indices[i]], float(scores[i])) for i in top_local]

    sources_text = "\n\n---\n\n".join([f"Record ID: {r['id']}\n{r['text']}" for r, _ in matches])
    user_prompt = f"[SOURCES]\n{sources_text}\n\n[USER QUESTION]\n{req.question}"

    # 3. Call active Groq Open-Source LLM (or fallback to Gemini 2.5 Flash)
    answer_text = None
    llm_used = None

    if groq_client and ACTIVE_GROQ_MODEL:
        try:
            completion = groq_client.chat.completions.create(
                model=ACTIVE_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            answer_text = completion.choices[0].message.content
            llm_used = f"groq/{ACTIVE_GROQ_MODEL}"
        except Exception as e:
            print(f"Groq generation error ({e}), falling back to Gemini...")

    if not answer_text:
        gem_res = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0
            )
        )
        answer_text = gem_res.text
        llm_used = "gemini/gemini-2.5-flash"

    fault_list = [
        MatchedFault(
            id=r["id"],
            make=r["make"],
            model=r["model"],
            system=r["system"],
            title=r["title"],
            severity=r["severity"],
            drive_advice=r["drive_advice"],
            obd_codes=r.get("obd_codes", []),
            score=round(score, 4)
        )
        for r, score in matches
    ]

    return DiagnoseResponse(
        answer=answer_text,
        llm_used=llm_used,
        matched_faults=fault_list
    )