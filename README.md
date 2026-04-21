# 🏥 Clinical Drug Safety Engine

A production-grade FastAPI backend for analyzing drug-drug interactions, allergy cross-reactivity, and condition-based contraindications. Built for healthcare-critical environments where incorrect outputs can harm patients.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                       (main.py)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  POST /analyze  ─────►  DrugSafetyEngine (engine.py)        │
│                         ┌─────────────────────────┐         │
│                         │ 1. Normalize Input      │         │
│                         │ 2. Check Cache          │         │
│                         │ 3. Run LLM Analysis     │         │
│                         │ 4. Validate LLM Output  │         │
│                         │ 5. Allergy Detection    │         │
│                         │ 6. Contraindications    │         │
│                         │ 7. Risk Scoring         │         │
│                         │ 8. Fallback if needed   │         │
│                         │ 9. Save Cache           │         │
│                         │ 10. Return Response     │         │
│                         └─────────────────────────┘         │
│                              │            │                 │
│                    ┌─────────┘            └────────┐        │
│                    ▼                               ▼        │
│            Medical LLM                     Fallback DB      │
│         (BioMistral-7B)              (fallback_interactions │
│          via llama-cpp                       .json)         │
│                                                             │
│              Cache Layer (cache.py)                          │
│         In-Memory (dict) │ Redis (optional)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🩺 Why a Medical-Specific LLM?

| Feature | Generic LLM (GPT, Claude) | Medical LLM (BioMistral) |
|---|---|---|
| **Medical Knowledge** | General, may hallucinate drugs | Trained on PubMed, medical texts |
| **Drug Interactions** | Unreliable for rare interactions | Aware of CYP450, pharmacokinetics |
| **Terminology** | May use lay terms | Uses precise pharmacological terms |
| **Hallucination Risk** | Higher for medical context | Lower — domain-specific training |
| **Data Privacy** | Requires API calls to external services | Runs locally — HIPAA-friendly |
| **Latency** | Network-dependent | Local inference, predictable |

**Supported Medical Models:**
- **BioMistral-7B** (recommended) — Medical fine-tune of Mistral-7B on PubMed
- **Med42** — Clinical reasoning model
- **Meditron** — Medical adaptation of LLaMA
- **OpenBioLLM** — Open-source biomedical LLM

All models run locally via `llama-cpp-python` with GGUF quantization for **<2GB VRAM**.

---

## 📁 Project Structure

```
evodocback/
├── main.py                          # FastAPI application & endpoints
├── engine.py                        # Core analysis pipeline (10-step)
├── models.py                        # Pydantic models (strict validation)
├── cache.py                         # Caching layer (in-memory + Redis)
├── prompts/
│   └── system_prompt.txt            # LLM system prompt (clinical expert)
├── data/
│   └── fallback_interactions.json   # 20+ real drug interactions dataset
├── tests/
│   └── test_engine.py               # Comprehensive test suite
├── requirements.txt                 # Pinned dependencies
├── .env.example                     # Configuration template
└── README.md                        # This file
```

---

## 🚀 Setup & Installation

### 1. Clone and Setup Environment

```bash
cd evodocback
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Download Medical LLM (Optional)

For LLM-powered analysis, download a GGUF-quantized medical model:

```bash
# Create models directory
mkdir -p models

# Download BioMistral-7B GGUF (recommended — ~4.4GB for Q4_K_M)
# From Hugging Face: https://huggingface.co/BioMistral/BioMistral-7B-GGUF
# Place the .gguf file in ./models/ and update MEDICAL_LLM_PATH in .env
```

> **Without an LLM model**, the engine runs in **fallback-only mode** using the curated interaction database. This is fully functional for known drug pairs.

### 4. Start the Server

```bash
# Development
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 📡 API Reference

### `POST /analyze`

Analyze drug interactions and patient safety.

**Request:**
```json
{
  "medicines": ["warfarin", "aspirin", "ibuprofen"],
  "patient_history": {
    "current_medications": ["metformin"],
    "allergies": ["penicillin"],
    "conditions": ["kidney disease"],
    "age": 65,
    "weight": 70
  }
}
```

**Response:**
```json
{
  "interactions": [
    {
      "drug_a": "warfarin",
      "drug_b": "aspirin",
      "severity": "high",
      "mechanism": "Both agents impair hemostasis through different mechanisms...",
      "clinical_recommendation": "Avoid concurrent use unless specifically indicated...",
      "source_confidence": "high"
    }
  ],
  "allergy_alerts": [],
  "contraindication_alerts": [
    {
      "medicine": "ibuprofen",
      "condition": "kidney disease",
      "risk_level": "high",
      "recommendation": "NSAIDs reduce renal blood flow..."
    }
  ],
  "safe_to_prescribe": false,
  "overall_risk_level": "high",
  "patient_risk_score": 85,
  "requires_doctor_review": true,
  "source": "fallback",
  "cache_hit": false,
  "processing_time_ms": 12.5
}
```

### `GET /health`

Health check with component status.

### `GET /docs`

Interactive Swagger UI documentation.

---

## 🗄️ Caching Strategy

| Feature | Detail |
|---|---|
| **Primary** | In-memory `dict` with TTL (default) |
| **Optional** | Redis backend (set `CACHE_BACKEND=redis`) |
| **Key** | `SHA-256(sorted(medicines) + sorted(current_medications))` |
| **TTL** | 1 hour (configurable via `CACHE_TTL`) |
| **Eviction** | Automatic on expiry, LRU-like at capacity |
| **Thread Safety** | `RLock` protected for concurrent access |
| **Max Size** | 10,000 entries (prevents memory bloat) |

Cache hits are indicated in the response via `cache_hit: true`.

---

## 🔄 Fallback Dataset

The file `data/fallback_interactions.json` contains:

- **20 clinically validated drug-drug interactions** with:
  - Pharmacological mechanisms
  - Severity ratings
  - Specific clinical recommendations
  - Source confidence levels

- **12 allergy cross-reactivity classes** mapping drug classes to individual drugs:
  - Penicillins, Cephalosporins, Sulfonamides, NSAIDs, Statins, ACE Inhibitors, ARBs, Fluoroquinolones, Opioids, Benzodiazepines, Tetracyclines, Macrolides

- **10 condition-contraindication rules** covering:
  - Kidney disease, Liver disease, Asthma, Diabetes, Heart failure, Pregnancy, Peptic ulcer, Hypothyroidism, Glaucoma, Hypertension

All data is sourced from established pharmacology references (Goodman & Gilman's, FDA drug labels, clinical guidelines).

---

## ⚡ Performance Evaluation

Tested on: MacBook Air (M1), 8GB RAM

| Scenario           | Time          |
| ------------------ | ------------- |
| Fallback (2 drugs) | ~5–10 ms      |
| Fallback (5 drugs) | ~10–30 ms     |
| LLM (5 drugs)      | ~1000–2500 ms |
| Cache hit          | <5 ms         |

Observations:

* Fallback mode is extremely fast and deterministic
* LLM adds latency but improves coverage
* Cache eliminates repeated computation almost entirely

Optimization Strategy:

* Cache-first design
* Minimal LLM calls
* Lightweight GGUF model for local inference

`processing_time_ms` is tracked and returned in every response.

---

## 🛡️ Safety & Reliability

This system is designed for healthcare-critical usage.

Key safeguards:

* No raw LLM output is exposed
* All outputs are schema-validated
* Hallucinated drugs are rejected
* Missing fields are filtered
* Fallback ensures non-empty response
* Doctor review required for uncertain cases

This ensures safe operation even under model failure conditions.

---

## 📋 Risk Scoring

| Event | Points |
|---|---|
| High interaction | +30 |
| Medium interaction | +15 |
| Low interaction | +5 |
| Critical allergy | +40 |
| Warning allergy | +15 |
| Contraindication | +25 |

**Risk Levels:**
- 0–30 → `low`
- 31–70 → `medium`
- 71–100 → `high`

Score is normalized to max 100.

---

## 🔧 Configuration

All settings are configurable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `MEDICAL_LLM_PATH` | _(empty)_ | Path to GGUF model file |
| `LLM_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `LLM_CONTEXT_LENGTH` | `2048` | Model context window |
| `LLM_TEMPERATURE` | `0.1` | Low temp for deterministic output |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🧩 Design Decisions

### 1. Hybrid LLM + Fallback Strategy

The system uses a hybrid approach:

* **LLM (BioMistral)** for flexible, contextual reasoning
* **Fallback dataset** for guaranteed safety and deterministic outputs

Tradeoff:

* LLM provides broader coverage but may be uncertain
* Fallback ensures reliability for known interactions

Decision:
Fallback is always available to prevent empty or unsafe responses.

---

### 2. Medical-Specific LLM Selection

BioMistral was chosen because:

* Trained on PubMed and clinical text
* Better understanding of pharmacokinetics (CYP450, metabolism)
* Lower hallucination risk compared to general LLMs

Tradeoff:

* Slightly slower than generic models
* Requires local inference setup

---

### 3. Local Inference (GGUF + llama.cpp)

Chosen for:

* Privacy (no external API calls)
* Low VRAM usage (<2GB)
* Works on clinic-grade hardware

Tradeoff:

* Reduced model size → slight drop in reasoning depth
* Requires prompt engineering to maintain accuracy

---

### 4. Caching Strategy

Key:
SHA-256(sorted(medicines) + sorted(current_medications))

Reason:

* Ensures deterministic cache hits
* Handles reordered inputs

Tradeoff:

* In-memory cache is fast but not distributed
* Redis support added for scalability

---

### 5. Strict Validation Layer

All LLM outputs are validated using Pydantic models.

Why:

* Prevent hallucinated drug names
* Enforce schema compliance
* Ensure medical safety

---

### 6. Safety-First Design

The system prioritizes patient safety:

* Any high-risk interaction → safe_to_prescribe = false
* Critical allergy → immediate rejection
* Low confidence → requires_doctor_review = true

This ensures the system never blindly trusts AI output.

---

## 📄 License

This project is intended for educational and research purposes. Not FDA-approved for clinical use without proper validation and regulatory review.
