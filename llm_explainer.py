"""
LLM Explanation Service — Vulnerability Analysis (Updated)
-----------------------------------------------------------
Changes from v1:
  - Dual LLM: CodeLlama 13B + DeepSeek-Coder 6.7B run in parallel
  - New field: remediated_code — clean C/C++ code extracted from remediation
  - New endpoint: /loop — feedback loop (original → fix → re-check → report)
  - New endpoint: /explain-file — accepts raw file content directly

Endpoints:
  POST /explain       — single code string, dual LLM analysis
  POST /explain-file  — same but accepts multipart file upload
  POST /loop          — full feedback loop: original → fix → re-check
  GET  /              — health check

Usage:
    uvicorn llm_explainer:app --host 0.0.0.0 --port 8001 --reload
"""

import re
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator

# ── Configuration ──────────────────────────────────────────────────────────────

OLLAMA_URL       = "http://localhost:11434/api/generate"
ROBERTA_URL      = "http://127.0.0.1:8000/predict"
MODEL_CODELLAMA  = "codellama:13b"
MODEL_DEEPSEEK   = "deepseek-coder:6.7b"
TEMPERATURE      = 0.2
MAX_TOKENS       = 1024
TIMEOUT_SECONDS  = 180

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Vulnerability Explainer API v2",
    description="Dual LLM (CodeLlama + DeepSeek-Coder) vulnerability analysis with feedback loop.",
    version="2.0.0"
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    code: str
    predicted_label: str
    confidence: float

    @field_validator('confidence', mode='before')
    @classmethod
    def parse_confidence(cls, v):
        return float(v)

    class Config:
        json_schema_extra = {
            "example": {
                "code": "int foo(char *buf) {\n    char tmp[10];\n    strcpy(tmp, buf);\n    return 0;\n}",
                "predicted_label": "non-vulnerable",
                "confidence": 0.5808
            }
        }

class LLMResult(BaseModel):
    llm_agreement: str
    explanation: str
    remediation: str
    remediated_code: str
    raw_response: str

class ExplainResponse(BaseModel):
    codellama: LLMResult
    deepseek: LLMResult

class LoopResponse(BaseModel):
    original_code: str
    round1_ml_label: str
    round1_ml_confidence: float
    round1_codellama: LLMResult
    round1_deepseek: LLMResult
    remediated_code: str
    round2_ml_label: str
    round2_ml_confidence: float
    round2_codellama: LLMResult
    round2_deepseek: LLMResult
    vulnerability_resolved: bool

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(code: str, predicted_label: str, confidence: float) -> str:
    confidence_pct = round(confidence * 100, 1)
    if predicted_label == "vulnerable":
        prediction_context = (
            f"An automated ML classifier (RoBERTa fine-tuned on DiverseVul) "
            f"has classified the following C/C++ function as VULNERABLE "
            f"with {confidence_pct}% confidence."
        )
    else:
        prediction_context = (
            f"An automated ML classifier (RoBERTa fine-tuned on DiverseVul) "
            f"has classified the following C/C++ function as NON-VULNERABLE "
            f"with {confidence_pct}% confidence."
        )

    return f"""You are an expert in software security specializing in C and C++ vulnerability analysis.

{prediction_context}

--- BEGIN FUNCTION ---
{code}
--- END FUNCTION ---

Your task is to perform a three-part analysis. Follow this exact structure:

## PART 1 - CONFIRMATION
State whether you AGREE or DISAGREE with the classifier's prediction.
Provide a one-sentence justification.

## PART 2 - EXPLANATION
(Only if vulnerable - either by classifier or your assessment.)
- Root cause: specific construct or pattern introducing the vulnerability
- Risky pattern: exact line(s) or operation(s) responsible
- Potential impact: what an attacker could achieve

If non-vulnerable and you agree, write: "No vulnerability identified."

## PART 3 - REMEDIATION
(Only if vulnerable.)
Provide ONLY the complete corrected C/C++ function inside a code block like this:
```c
// corrected function here
```
Then briefly explain what was changed and why.

If non-vulnerable and you agree, write: "No remediation needed."

Be technical, precise, and concise."""

# ── Ollama caller ──────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"{model} timed out. Try again.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Run: ollama serve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error ({model}): {str(e)}")

# ── Code extractor ─────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    match = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# ── Response parser ────────────────────────────────────────────────────────────

def parse_response(raw: str) -> LLMResult:
    agreement = "unknown"
    explanation = ""
    remediation = ""
    remediated_code = ""

    raw_upper = raw.upper()
    if "DISAGREE" in raw_upper:
        agreement = "disagrees"
    elif "AGREE" in raw_upper:
        agreement = "agrees"

    sections = raw.split("## PART")
    for section in sections:
        s = section.strip()
        if s.startswith("2"):
            explanation = s.split("\n", 1)[-1].strip()
        elif s.startswith("3"):
            remediation = s.split("\n", 1)[-1].strip()
            remediated_code = extract_code_block(remediation)

    if not explanation and not remediation:
        explanation = raw

    return LLMResult(
        llm_agreement=agreement,
        explanation=explanation or "No explanation provided.",
        remediation=remediation or "No remediation provided.",
        remediated_code=remediated_code,
        raw_response=raw
    )

# ── RoBERTa caller ─────────────────────────────────────────────────────────────

def call_roberta(code: str) -> dict:
    try:
        response = requests.post(ROBERTA_URL, json={"code": code}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to RoBERTa API on port 8000.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RoBERTa error: {str(e)}")

# ── Core dual LLM function ─────────────────────────────────────────────────────

def run_dual_llm(code: str, predicted_label: str, confidence: float) -> ExplainResponse:
    prompt = build_prompt(code, predicted_label, confidence)
    raw_codellama = call_ollama(prompt, MODEL_CODELLAMA)
    raw_deepseek  = call_ollama(prompt, MODEL_DEEPSEEK)
    return ExplainResponse(
        codellama=parse_response(raw_codellama),
        deepseek=parse_response(raw_deepseek)
    )

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "message": "LLM Explainer API v2 running.",
        "models": [MODEL_CODELLAMA, MODEL_DEEPSEEK]
    }


@app.post("/explain", response_model=ExplainResponse, summary="Dual LLM analysis from code string")
def explain(request: ExplainRequest):
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty.")
    if request.predicted_label not in ("vulnerable", "non-vulnerable"):
        raise HTTPException(status_code=400, detail="predicted_label must be 'vulnerable' or 'non-vulnerable'.")
    return run_dual_llm(request.code, request.predicted_label, request.confidence)


@app.post("/explain-file", response_model=ExplainResponse, summary="Dual LLM analysis from .c/.cpp file upload")
async def explain_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.c', '.cpp')):
        raise HTTPException(status_code=400, detail="Only .c or .cpp files are accepted.")
    content = await file.read()
    code = content.decode('utf-8', errors='replace')
    if not code.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ml_result = call_roberta(code)
    return run_dual_llm(code, ml_result["label"], ml_result["confidence"])


@app.post("/loop", response_model=LoopResponse, summary="Feedback loop: original code → fix → re-check")
def loop(request: ExplainRequest):
    """
    Round 1: Analyze original code with RoBERTa + dual LLM
    Extract: Get remediated code from CodeLlama (fallback to DeepSeek)
    Round 2: Re-analyze the fixed code with RoBERTa + dual LLM
    Report: Was the vulnerability resolved?
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty.")

    # Round 1
    print("[LOOP] Round 1 - analyzing original code...")
    round1_llm = run_dual_llm(request.code, request.predicted_label, request.confidence)

    # Extract remediated code
    remediated_code = round1_llm.codellama.remediated_code
    if not remediated_code:
        remediated_code = round1_llm.deepseek.remediated_code
    if not remediated_code:
        print("[LOOP] Warning: no remediated code extracted, using original for round 2")
        remediated_code = request.code

    # Round 2
    print("[LOOP] Round 2 - re-checking remediated code...")
    round2_ml = call_roberta(remediated_code)
    round2_llm = run_dual_llm(remediated_code, round2_ml["label"], round2_ml["confidence"])

    # Resolved = ML predicts non-vulnerable AND both LLMs agree
    vulnerability_resolved = (
        round2_ml["label"] == "non-vulnerable" and
        round2_llm.codellama.llm_agreement == "agrees" and
        round2_llm.deepseek.llm_agreement == "agrees"
    )

    return LoopResponse(
        original_code=request.code,
        round1_ml_label=request.predicted_label,
        round1_ml_confidence=request.confidence,
        round1_codellama=round1_llm.codellama,
        round1_deepseek=round1_llm.deepseek,
        remediated_code=remediated_code,
        round2_ml_label=round2_ml["label"],
        round2_ml_confidence=round2_ml["confidence"],
        round2_codellama=round2_llm.codellama,
        round2_deepseek=round2_llm.deepseek,
        vulnerability_resolved=vulnerability_resolved
    )
