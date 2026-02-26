"""
LLM Explanation Service — Vulnerability Analysis via CodeLlama
--------------------------------------------------------------
This is the second service in the pipeline. It receives:
  - The original C/C++ function
  - The RoBERTa model prediction (label + confidence)

And returns a structured three-part analysis:
  1. Confirmation (does the LLM agree with the ML prediction?)
  2. Explanation (root cause, risky pattern, potential impact)
  3. Remediation (corrected code or precise fix description)

Exposes a FastAPI endpoint on port 8001.

Usage:
    uvicorn llm_explainer:app --host 0.0.0.0 --port 8001 --reload
"""

import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── Configuration ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:13b"

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Vulnerability Explainer API",
    description=(
        "Receives a C/C++ function and a RoBERTa prediction, then uses "
        "CodeLlama to confirm, explain, and propose a fix for the vulnerability."
    ),
    version="1.0.0"
)

# ── Request / Response schemas ─────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    code: str               # The raw C/C++ function
    predicted_label: str    # "vulnerable" or "non-vulnerable"
    confidence: float       # e.g. 0.9823 — accepts string or float

    @field_validator('confidence', mode='before')
    @classmethod
    def parse_confidence(cls, v):
        return float(v)

    class Config:
        json_schema_extra = {
            "example": {
                "code": "int foo(char *buf) {\n    char tmp[10];\n    strcpy(tmp, buf);\n    return 0;\n}",
                "predicted_label": "vulnerable",
                "confidence": 0.9823
            }
        }

class ExplainResponse(BaseModel):
    llm_agreement: str       # "agrees" or "disagrees"
    explanation: str         # Root cause, risky pattern, impact
    remediation: str         # Fix description or corrected code
    raw_response: str        # Full unprocessed LLM output (useful for debugging)

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

    prompt = f"""You are an expert in software security specializing in C and C++ vulnerability analysis.

{prediction_context}

--- BEGIN FUNCTION ---
{code}
--- END FUNCTION ---

Your task is to perform a three-part analysis. Follow this exact structure in your response:

## PART 1 — CONFIRMATION
State whether you AGREE or DISAGREE with the classifier's prediction.
Provide a one-sentence justification for your assessment.

## PART 2 — EXPLANATION
(Complete this part only if the function is vulnerable, either because the classifier predicted it or because you disagree and believe it IS vulnerable.)
Provide a technical explanation covering:
- Root cause: what specific construct or pattern introduces the vulnerability
- Risky pattern: the exact line(s) or operation(s) responsible
- Potential impact: what an attacker could achieve by exploiting this

If the function is non-vulnerable and you agree with the classifier, write: "No vulnerability identified."

## PART 3 — REMEDIATION
(Complete this part only if the function is vulnerable.)
Propose a concrete fix. Either:
- Provide a corrected version of the function in C/C++, OR
- Describe precisely what changes must be made and why

If the function is non-vulnerable and you agree with the classifier, write: "No remediation needed."

Be technical, precise, and concise. Do not restate the classifier's output — provide your own independent expert assessment.
"""
    return prompt

# ── Ollama caller ──────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,        # Wait for full response before returning
        "options": {
            "temperature": 0.2, # Low temperature = more deterministic, better for technical analysis
            "num_predict": 1024 # Max tokens in response
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out. The model may be slow on first run — try again."
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure it is running: 'ollama serve'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

# ── Response parser ────────────────────────────────────────────────────────────

def parse_response(raw: str, predicted_label: str) -> dict:
    """
    Extracts the three sections from the LLM response.
    Falls back gracefully if the LLM doesn't follow the format perfectly.
    """
    agreement = "unknown"
    explanation = ""
    remediation = ""

    raw_upper = raw.upper()

    # Detect agreement
    if "AGREE" in raw_upper and "DISAGREE" not in raw_upper:
        agreement = "agrees"
    elif "DISAGREE" in raw_upper:
        agreement = "disagrees"

    # Extract sections by splitting on headers
    sections = raw.split("## PART")
    for section in sections:
        section_stripped = section.strip()
        if section_stripped.startswith("2"):
            explanation = section_stripped.split("\n", 1)[-1].strip()
        elif section_stripped.startswith("3"):
            remediation = section_stripped.split("\n", 1)[-1].strip()

    # Fallback: if parsing fails, return the full response in explanation
    if not explanation and not remediation:
        explanation = raw
        remediation = "Could not parse remediation — see full response."

    return {
        "llm_agreement": agreement,
        "explanation": explanation or "No explanation provided.",
        "remediation": remediation or "No remediation provided.",
        "raw_response": raw
    }

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "LLM Explainer API is running.", "model": MODEL_NAME}


@app.post("/explain", response_model=ExplainResponse, summary="Explain vulnerability prediction")
def explain(request: ExplainRequest):
    """
    Accepts a C/C++ function and a RoBERTa prediction, returns a
    structured three-part LLM analysis: confirmation, explanation, remediation.
    """
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty.")

    if request.predicted_label not in ("vulnerable", "non-vulnerable"):
        raise HTTPException(
            status_code=400,
            detail="predicted_label must be 'vulnerable' or 'non-vulnerable'."
        )

    prompt = build_prompt(request.code, request.predicted_label, request.confidence)
    raw_response = call_ollama(prompt)
    parsed = parse_response(raw_response, request.predicted_label)

    return ExplainResponse(**parsed)


@app.post("/analyze", summary="Full pipeline: predict + explain in one call")
def analyze(code: str):
    """
    Convenience endpoint: calls the RoBERTa API on port 8000 first,
    then automatically passes the result to the explainer.
    Useful for testing the full pipeline without N8N.
    """
    # Step 1: call RoBERTa API
    try:
        ml_response = requests.post(
            "http://localhost:8000/predict",
            json={"code": code},
            timeout=30
        )
        ml_response.raise_for_status()
        ml_result = ml_response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to RoBERTa API on port 8000. Make sure it is running."
        )

    # Step 2: call LLM explainer
    explain_request = ExplainRequest(
        code=code,
        predicted_label=ml_result["label"],
        confidence=ml_result["confidence"]
    )
    prompt = build_prompt(code, ml_result["label"], ml_result["confidence"])
    raw_response = call_ollama(prompt)
    parsed = parse_response(raw_response, ml_result["label"])

    # Return combined result
    return {
        "ml_prediction": ml_result,
        "llm_analysis": parsed
    }
