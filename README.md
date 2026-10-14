# Car Diagnostic RAG Microservice 🚗🔧

High-performance automotive diagnostic microservice built with **FastAPI**, **Gemini Multilingual Embeddings**, and **Groq Open-Source LLMs** (with automatic **Google Gemini 2.5 Flash** fallback).

Designed for low latency, zero cold-start re-indexing, multilingual query support (Arabic & English), and seamless integration into .NET backend architectures.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites & Environment Variables](#prerequisites--environment-variables)
- [How to Run](#how-to-run)
  - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
  - [Option B: Local Python Virtual Environment](#option-b-local-python-virtual-environment)
- [Dataset & Re-Indexing Workflow](#dataset--re-indexing-workflow)
- [API Reference](#api-reference)
  - [GET /health](#get-health)
  - [POST /api/v1/diagnose](#post-apiv1diagnose)
  - [GET /docs](#get-docs)
  - [GET /demo](#get-demo)
- [.NET Backend Integration Guide](#net-backend-integration-guide)

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       .NET Backend / Web API          |
                      +---------------------------------------+
                                          |
                                          | HTTP POST /api/v1/diagnose
                                          v
+---------------------------------------------------------------------------------+
|                         Car Diagnostic RAG Microservice                         |
|                                                                                 |
| 1. Query Embedder: Gemini text-embedding-001 (Multilingual: Arabic + English)   |
| 2. Vector Search: Pre-loaded NumPy Cosine Similarity + OBD-II Keyword Boost     |
| 3. Source Injection: Top-K domain fault records injected into strict prompt     |
| 4. Primary LLM: Groq Open-Source Models (Llama 3.3 / Qwen / DeepSeek)           |
| 5. Secondary LLM (Fallback): Gemini 2.5 Flash if Groq is unavailable/throttled  |
+---------------------------------------------------------------------------------+
```

### Key Technical Characteristics:
1. **Zero Cold-Start Lag**: Embedding vectors (`vectors.npy`) and structured metadata (`records.json`) are pre-computed and bundled. The service loads them into RAM in milliseconds on startup.
2. **Dynamic Groq Model Selection**: At boot, the service inspects your Groq account for available high-tier open-source chat models (prioritizing Llama, Qwen, DeepSeek, Gemma, Mixtral) and selects the best active model automatically.
3. **Resilient Dual-LLM Pipeline**: If Groq encounters API rate limits or network issues, queries automatically fall back to Gemini 2.5 Flash without throwing an error to the caller.
4. **OBD-II Hybrid Boosting**: If the user's inquiry contains an OBD-II code (e.g. `P0300`, `P0420`), records mentioning that code receive an automatic similarity score boost (+0.5).
5. **Strict Grounding**: The system prompt enforces strict mechanical grounding on retrieved sources only and responds in the user's language (Arabic or English).

---

## Repository Structure

```
car-rag-service/
├── .env.example                # Example environment variables file
├── .gitignore                  # Git ignore rules (keeps vector & record caches)
├── .dockerignore               # Docker ignore rules
├── Dockerfile                  # Container definition with health checks
├── docker-compose.yml          # 1-command Docker orchestration
├── requirements.txt            # Python dependencies
├── app.py                      # FastAPI application & RAG query engine
├── ingest.py                   # Data ingestion and vector embedding script
├── byd_f3_faults_en.jsonl      # Source automotive fault records (JSONL format)
├── records.json                # Pre-processed records for fast runtime lookup
├── vectors.npy                 # Pre-computed 768-d normalized embeddings
├── demo.html                   # Interactive browser UI for manual testing
└── dotnet-integration-sample/  # Ready-to-copy C# code for the .NET team
    ├── CarDiagnosticDtos.cs    # Strongly-typed C# records matching JSON schema
    └── CarDiagnosticClient.cs  # Typed HttpClient implementation & DI setup
```

---

## Prerequisites & Environment Variables

Create your local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Populate the required API keys in `.env`:

```ini
# Required: API key for Groq's high-speed inference engine
GROQ_API_KEY=gsk_your_groq_api_key_here

# Required: API key for Google Gemini (Embeddings + LLM Fallback)
GEMINI_API_KEY=AIza_your_gemini_api_key_here
```

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Used for ultra-fast generation with open-source LLMs. | Yes |
| `GEMINI_API_KEY` | Used for query vector embeddings (`gemini-embedding-001`) and fallback chat (`gemini-2.5-flash`). | Yes |

---

## How to Run

### Option A: Docker Compose (Recommended)

Run the service as an isolated container in one command:

```bash
docker compose up -d --build
```

- **Service Port:** `http://localhost:8001`
- **Container Health Check:** Monitored automatically via `curl -f http://localhost:8001/health`
- **Logs:** View logs using `docker compose logs -f`
- **Shutdown:** `docker compose down`

### Option B: Local Python Virtual Environment

If running directly on a host machine without Docker:

```bash
# 1. Create and activate a Python 3.11+ virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Uvicorn server
uvicorn app:app --host 0.0.0.0 --port 8001
```

---

## Dataset & Re-Indexing Workflow

The service uses pre-computed vectors and metadata. You do **not** need to re-index unless the underlying automotive knowledge base changes.

### When Adding New Car Models or Faults:
1. Update or append new entries to the dataset JSONL file (e.g. `byd_f3_faults_en.jsonl`).
   Each entry must adhere to this JSON structure:
   ```json
   {
     "id": "byd-f3-031",
     "language": "en",
     "make": "BYD",
     "model": "F3",
     "system": "Cooling",
     "title": "Engine overheating at idle",
     "symptoms": ["Temperature gauge rising in traffic", "Radiator fan not turning on"],
     "possible_causes": ["Faulty radiator fan relay", "Defective coolant temperature sensor"],
     "diagnosis_steps": ["Check fuse 15", "Test fan motor with direct 12V"],
     "fix": "Replace fan relay or fan motor assembly.",
     "severity": "high",
     "drive_advice": "Do not drive if gauge enters red zone; pull over immediately.",
     "obd_codes": ["P0117", "P0118"],
     "text": "Make: BYD | Model: F3 | System: Cooling | Title: Engine overheating at idle..."
   }
   ```
2. Run the ingestion script:
   ```bash
   python ingest.py
   ```
   This will:
   - Read each line from the `.jsonl` file.
   - Batch-embed texts using Google Gemini's `gemini-embedding-001` model.
   - Normalize and save vectors to `vectors.npy`.
   - Save the formatted JSON records to `records.json`.
3. If using Docker, rebuild the image to include the new binary index:
   ```bash
   docker compose up -d --build
   ```

---

## API Reference

### GET /health
Used for container orchestration, Kubernetes probes, and load balancer health checks.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "records_loaded": 30,
  "vectors_ready": true
}
```

---

### POST /api/v1/diagnose
Main endpoint for querying automotive diagnostic intelligence.

**Request Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "make": "BYD",
  "model": "F3",
  "question": "العربية بتنتش وبترج على السلانسيه ولمبة الموتور بتنور، كود P0300",
  "top_k": 2
}
```

#### Request Fields
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `make` | string | Optional | `"BYD"` | Target manufacturer name. |
| `model` | string | Optional | `"F3"` | Target model name. |
| `question` | string | **Yes** | — | User's question or problem description (Arabic or English). |
| `top_k` | integer | Optional | `2` | Number of most relevant fault records to retrieve. |

**Response:** `200 OK`
```json
{
  "answer": "بناءً على السجلات المتاحة:\n\nالعطل المرجح:\nوجود خلل في الإشعال أو تشغيل خشن للمحرك (Engine Misfire).\n\nالأسباب المحتملة:\n1. تآكل أو ضبط غير سليم للبوجيهات (Spark Plugs).\n2. ضعف في موبينات الإشعال (Ignition Coils).\n3. تسريب هواء أو خلل بحساس الهواء (MAF).\n\nخطوات الفحص:\n1. فحص البوجيهات والتأكد من حالتها.\n2. تبديل مكان الموبينة للتأكد من انتقال العطل معها.\n\nالإصلاح:\nتغيير البوجيهات التالفة بالنوع الموصى به واستبدال الموبينة المعطوبة.\n\nهل يمكن القيادة بها:\nيمكن القيادة لمسافة قصيرة إذا كانت لمبة الأعطال ثابتة، ولكن تجنب القيادة تماماً إذا كانت اللمبة تومض (Flashing) لتفادي تلف علبة البيئة.",
  "llm_used": "groq/llama-3.3-70b-versatile",
  "matched_faults": [
    {
      "id": "byd-f3-001",
      "make": "BYD",
      "model": "F3",
      "system": "Engine",
      "title": "Engine misfire and rough running",
      "severity": "medium",
      "drive_advice": "Short distances are OK if the light is steady; avoid driving if it is flashing because it can damage the catalytic converter",
      "obd_codes": [
        "P0300",
        "P0301",
        "P0302",
        "P0303",
        "P0304"
      ],
      "score": 1.3412
    },
    {
      "id": "byd-f3-005",
      "make": "BYD",
      "model": "F3",
      "system": "Fuel System",
      "title": "Clogged fuel injectors causing hesitation",
      "severity": "medium",
      "drive_advice": "Car can be driven to a workshop but avoid aggressive throttle",
      "obd_codes": [
        "P0171"
      ],
      "score": 0.6587
    }
  ]
}
```

#### Response Fields
| Field | Type | Description |
|---|---|---|
| `answer` | string | Strict AI diagnosis formatted in the language of the question (Arabic or English). |
| `llm_used` | string | Identifier of the LLM that generated the answer (e.g. `groq/llama-3.3-70b-versatile` or `gemini/gemini-2.5-flash`). |
| `matched_faults` | array | Array of matched records retrieved from vector similarity search. |
| `matched_faults[].id` | string | Unique record ID in the knowledge base. |
| `matched_faults[].make` | string | Vehicle make. |
| `matched_faults[].model` | string | Vehicle model. |
| `matched_faults[].system` | string | Vehicle subsystem (e.g., `Engine`, `Braking`, `Cooling`, `Transmission`). |
| `matched_faults[].title` | string | Title/summary of the fault. |
| `matched_faults[].severity` | string | Severity rating (`low`, `medium`, `high`, `critical`). |
| `matched_faults[].drive_advice`| string | Operational safety advice (can the vehicle be driven safely?). |
| `matched_faults[].obd_codes` | string[] | Related OBD-II diagnostic trouble codes. |
| `matched_faults[].score` | number | Relevance score (cosine similarity score + OBD code match boost). |

#### HTTP Status Codes
| Status | Meaning | Description |
|---|---|---|
| `200 OK` | Success | Diagnostic processed and answered successfully. |
| `404 Not Found` | Unknown Make/Model | No fault records exist in the index for the specified `make` and `model`. |
| `422 Unprocessable Entity` | Validation Error | Request body missing `question` or contains invalid parameters. |
| `500 Internal Server Error` | Server Error | Unhandled backend exception. |

---

### GET /docs
Interactive OpenAPI documentation (Swagger UI).
- URL: `http://localhost:8001/docs`

### GET /demo
Interactive HTML/JS test console bundled with the service.
- URL: `http://localhost:8001/demo` or `http://localhost:8001/`

---

## .NET Backend Integration Guide

Ready-to-use C# integration code is located in the [`dotnet-integration-sample/`](dotnet-integration-sample/) directory:

- [`CarDiagnosticDtos.cs`](dotnet-integration-sample/CarDiagnosticDtos.cs): C# records with `[JsonPropertyName]` attributes matching the API schema.
- [`CarDiagnosticClient.cs`](dotnet-integration-sample/CarDiagnosticClient.cs): Typed `HttpClient` implementing `ICarDiagnosticClient`.

### 1. Copy Files to Your .NET Project
Copy both files from `dotnet-integration-sample/` into your C# solution (e.g. inside `Services/CarRag/` or `Clients/`).

### 2. Configure `appsettings.json`
Add the microservice base URL to your configuration:

```json
{
  "CarRagService": {
    "BaseUrl": "http://localhost:8001"
  }
}
```

### 3. Register in `Program.cs`
In your ASP.NET Core `Program.cs` (.NET 6 / 7 / 8 / 9):

```csharp
using CarRagService.Client;

var builder = WebApplication.CreateBuilder(args);

// Register Car Diagnostic Typed HttpClient
builder.Services.AddHttpClient<ICarDiagnosticClient, CarDiagnosticClient>(client =>
{
    var baseUrl = builder.Configuration["CarRagService:BaseUrl"] ?? "http://localhost:8001";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(60);
});

var app = builder.Build();
```

### 4. Inject and Query in a Controller or Minimal API

```csharp
app.MapPost("/api/car/diagnose", async (
    ICarDiagnosticClient diagnosticClient, 
    DiagnoseRequestDto request, 
    CancellationToken ct) =>
{
    try
    {
        DiagnoseResponseDto response = await diagnosticClient.DiagnoseAsync(request, ct);
        return Results.Ok(response);
    }
    catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
    {
        return Results.NotFound(new { message = "Unsupported vehicle make/model." });
    }
    catch (Exception ex)
    {
        return Results.Problem(detail: ex.Message, statusCode: 500);
    }
});
```

---

## Production Handoff Checklist
- [x] Pre-computed index files (`records.json`, `vectors.npy`) present and tracked in Git.
- [x] Multi-language embedding model configured (`gemini-embedding-001`).
- [x] Primary open-source generation configured on Groq with automated model selection.
- [x] Automatic fallback to Gemini 2.5 Flash on Groq failure.
- [x] `/health` endpoint available for container healthchecks.
- [x] Dockerfile & Docker Compose configured on port `8001`.
- [x] Typed C# DTOs and client prepared for the .NET team.
