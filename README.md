# 🛡️ DiverseVul — Vulnerability Detection & Explanation Pipeline

A two-stage pipeline for detecting vulnerabilities in C/C++ source code:

1. **ML Classification** — A fine-tuned RoBERTa model predicts whether a function is *vulnerable* or *non-vulnerable*
2. **LLM Explanation** — CodeLlama analyzes the prediction, provides a technical explanation, and suggests remediation

Built on the [DiverseVul](https://github.com/wagner-group/diversevul) dataset (~330K real-world C/C++ functions from 150+ CVEs).

---

## 📁 Repository Structure

```
fl_DiverseVul/
├── app.py                    # FastAPI service for RoBERTa inference (port 8000)
├── llm_explainer.py          # FastAPI service for CodeLlama explanation (port 8001)
├── run_evaluation.py         # Batch evaluation script (50 balanced samples)
├── evaluation_results.csv    # Saved evaluation results
├── requirements.txt          # Python dependencies
├── models/
│   ├── roberta/
│   │   └── notebook.ipynb    # Training notebook (RoBERTa fine-tuning)
│   ├── codebert/
│   │   └── codebert.ipynb    # CodeBERT experiments
│   └── t5                    # T5 experiments (placeholder)
├── CodeBERT.pdf              # Reference paper
└── GraphCodeBERT.pdf         # Reference paper
```

---

## 🏗️ Architecture

```
┌────────────┐     POST /predict     ┌──────────────────┐
│  C/C++     │ ──────────────────►   │  RoBERTa API     │
│  Function  │                       │  (port 8000)     │
└────────────┘                       └───────┬──────────┘
                                             │ prediction + confidence
                                             ▼
                                     ┌──────────────────┐
                                     │  LLM Explainer   │
                                     │  (port 8001)     │
                                     │  CodeLlama 13B   │
                                     └───────┬──────────┘
                                             │
                                             ▼
                                     ┌──────────────────┐
                                     │  Structured      │
                                     │  Response:       │
                                     │  • Confirmation  │
                                     │  • Explanation   │
                                     │  • Remediation   │
                                     └──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (for CodeLlama)
- Model checkpoint in `./results/checkpoint-49575/`

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the RoBERTa API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the LLM Explainer

Make sure Ollama is running with CodeLlama:

```bash
ollama pull codellama:13b
ollama serve
```

Then start the explainer service:

```bash
uvicorn llm_explainer:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Test a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "int foo(char *buf) {\n    char tmp[10];\n    strcpy(tmp, buf);\n    return 0;\n}"}'
```

---

## 🔬 API Endpoints

### RoBERTa API (`port 8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Health check |
| `POST` | `/predict` | Predict vulnerability from C/C++ function |

**POST `/predict`** — Request:
```json
{ "code": "int foo(char *buf) { ... }" }
```

**Response:**
```json
{
  "label": "vulnerable",
  "label_id": 1,
  "confidence": 0.9823,
  "probabilities": {
    "non-vulnerable": 0.0177,
    "vulnerable": 0.9823
  }
}
```

### LLM Explainer API (`port 8001`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`        | Health check |
| `POST` | `/explain` | Explain a RoBERTa prediction |
| `POST` | `/analyze` | Full pipeline in one call (predict + explain) |

---

## 🧠 Model Details

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `roberta-base` |
| Task | Binary classification (vulnerable vs. non-vulnerable) |
| Dataset | DiverseVul (2023-07-02 snapshot) |
| Total samples | 330,492 |
| Train / Val / Test | 264,393 / 33,049 / 33,050 (80/10/10, stratified) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 5e-5 (default) |
| Optimizer | AdamW |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max token length | 512 |
| Class weights | `[0.5304, 8.7224]` (sklearn `balanced`) |
| Training time | ~2h 36min |

### Class Imbalance Handling

The dataset is heavily imbalanced. A custom `Trainer` overrides `compute_loss()` to use `nn.CrossEntropyLoss` with class weights computed via `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`.

---

## 📊 Running Batch Evaluation

The evaluation script samples 50 balanced examples (25 vulnerable + 25 non-vulnerable) and runs them through both services:

```bash
python run_evaluation.py --dataset /path/to/diversevul_20230702.json
```

To skip the LLM step (ML predictions only):

```bash
python run_evaluation.py --dataset /path/to/diversevul_20230702.json --skip-llm
```

Results are saved to `evaluation_results.csv`.

---

## 📚 References

- [DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection](https://github.com/wagner-group/diversevul)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [CodeLlama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)