"""
FastAPI wrapper for RoBERTa Vulnerability Detection Model
---------------------------------------------------------
Exposes a local REST API endpoint that accepts a C/C++ function
and returns a binary vulnerability prediction.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then POST to: http://localhost:8000/predict
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "./results/checkpoint-49575" 

MAX_LEN = 512

LABEL_MAP = {
    0: "non-vulnerable",
    1: "vulnerable"
}

# ── Load model & tokenizer at startup ─────────────────────────────────────────

print(f"Loading model from: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found at '{CHECKPOINT_PATH}'.\n"
        f"Run 'ls ./results/' to find your checkpoint folder name "
        f"and update CHECKPOINT_PATH in app.py."
    )

# mps = torch.device("mps")
cpu = torch.device("cpu")
device=cpu
print(f"Using device: {device}")

tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT_PATH)
model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.to(device)
model.eval()

print("Model and tokenizer loaded successfully.")

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RoBERTa Vulnerability Detection API",
    description=(
        "Accepts a C/C++ function as input and returns a binary prediction: "
        "vulnerable or non-vulnerable. Designed as the ML layer in a "
        "vulnerability detection + LLM explanation pipeline."
    ),
    version="1.0.0"
)

# ── Request / Response schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    code: str  # The raw C/C++ function as a string

    class Config:
        json_schema_extra = {
            "example": {
                "code": "int foo(char *buf) {\n    char tmp[10];\n    strcpy(tmp, buf);\n    return 0;\n}"
            }
        }

class PredictResponse(BaseModel):
    label: str          # "vulnerable" or "non-vulnerable"
    label_id: int       # 1 or 0
    confidence: float   # probability of the predicted class (0.0 - 1.0)
    probabilities: dict # {"non-vulnerable": float, "vulnerable": float}

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "RoBERTa Vulnerability Detection API is running."}


@app.post("/predict", response_model=PredictResponse, summary="Predict vulnerability")
def predict(request: PredictRequest):
    """
    Accepts a C/C++ function string and returns:
    - **label**: 'vulnerable' or 'non-vulnerable'
    - **label_id**: 1 (vulnerable) or 0 (non-vulnerable)
    - **confidence**: probability score for the predicted class
    - **probabilities**: full softmax distribution over both classes
    """
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty.")

    # Tokenize — mirrors your training CodeDataset exactly
    encoding = tokenizer.encode_plus(
        request.code,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: [1, 2]

    probs = F.softmax(logits, dim=-1).squeeze()  # shape: [2]
    predicted_id = torch.argmax(probs).item()
    confidence = probs[predicted_id].item()

    return PredictResponse(
        label=LABEL_MAP[predicted_id],
        label_id=predicted_id,
        confidence=round(confidence, 4),
        probabilities={
            "non-vulnerable": round(probs[0].item(), 4),
            "vulnerable": round(probs[1].item(), 4)
        }
    )