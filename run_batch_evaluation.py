"""
run_batch_evaluation.py
-----------------------
Sends each function from test_batch.json through the full pipeline:

    RoBERTa (POST /predict, port 8000)
        → LLM Explainer (POST /explain, port 8001)
            → Gemini judge if CodeLlama & DeepSeek disagree

Matches your actual llm_explainer.py v2 API exactly:
  - /explain expects: { code, predicted_label, confidence }
  - /explain returns: { codellama: LLMResult, deepseek: LLMResult }
  - LLMResult fields: llm_agreement ("agrees"/"disagrees"),
                      explanation, remediation, remediated_code, raw_response
  - Verdict is INFERRED from llm_agreement + roberta label (not a direct field)

Usage:
    python run_batch_evaluation.py --input test_batch.json --output evaluation_results.csv

Prerequisites (must already be running):
    Terminal 1: uvicorn app:app --host 0.0.0.0 --port 8000
    Terminal 2: uvicorn llm_explainer:app --host 0.0.0.0 --port 8001
    Terminal 3: ollama serve
    Env var:    export GEMINI_API_KEY="your_key_here"  (only needed for tie-breaking)
"""

import json
import csv
import time
import argparse
import os
import requests
from pathlib import Path

# ── Endpoint config ────────────────────────────────────────────────────────────
ROBERTA_URL       = "http://localhost:8000/predict"
LLM_EXPLAINER_URL = "http://localhost:8001/explain"
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

REQUEST_TIMEOUT   = 180   # seconds (LLMs can be slow)
SLEEP_BETWEEN     = 1.0   # seconds between samples
# ───────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "index", "cwe_sampled", "ground_truth",
    "roberta_label", "roberta_confidence",
    "codellama_agreement",   # "agrees" / "disagrees" / "unknown"
    "deepseek_agreement",    # "agrees" / "disagrees" / "unknown"
    "codellama_verdict",     # derived: what CodeLlama actually thinks
    "deepseek_verdict",      # derived: what DeepSeek actually thinks
    "llm_agreement",         # True if both LLMs reach same verdict
    "gemini_judge_verdict",  # filled only on LLM disagreement
    "final_verdict",         # pipeline's final answer
    "correct",               # True if final_verdict == ground_truth
    "remediation_provided",  # True if any fix code was extracted
    "latency_s",
    "error",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def call_roberta(code: str) -> dict:
    """Returns: { label, label_id, confidence, probabilities }"""
    r = requests.post(ROBERTA_URL, json={"code": code}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_llm_explainer(code: str, roberta_label: str, roberta_conf: float) -> dict:
    """
    POST /explain with { code, predicted_label, confidence }
    Returns: { codellama: LLMResult, deepseek: LLMResult }
    """
    payload = {
        "code": code,
        "predicted_label": roberta_label,
        "confidence": roberta_conf,
    }
    r = requests.post(LLM_EXPLAINER_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def derive_verdict(llm_agreement: str, roberta_label: str) -> str:
    """
    Your LLM returns 'agrees' or 'disagrees' relative to RoBERTa.
    This converts that into an absolute verdict.
    """
    if llm_agreement == "agrees":
        return roberta_label
    elif llm_agreement == "disagrees":
        return "non-vulnerable" if roberta_label == "vulnerable" else "vulnerable"
    else:
        # Unknown — fall back to RoBERTa
        return roberta_label


def call_gemini_judge(code: str, cl_explanation: str, ds_explanation: str,
                      cl_verdict: str, ds_verdict: str) -> str:
    """Ask Gemini to break the LLM tie. Returns 'vulnerable' or 'non-vulnerable'."""
    if not GEMINI_API_KEY:
        # No key: fall back to CodeLlama's verdict
        return cl_verdict

    prompt = (
        "You are a security expert. Two models disagreed on this C/C++ function.\n\n"
        f"CodeLlama verdict: {cl_verdict}\n"
        f"CodeLlama reasoning: {cl_explanation[:400]}\n\n"
        f"DeepSeek verdict: {ds_verdict}\n"
        f"DeepSeek reasoning: {ds_explanation[:400]}\n\n"
        "Code:\n```c\n" + code[:1500] + "\n```\n\n"
        "Respond with ONLY one word: 'vulnerable' or 'non-vulnerable'."
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    return "non-vulnerable" if "non" in text else "vulnerable"


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(batch: list[dict], out_csv: Path):
    total = len(batch)
    results = []

    print(f"\nStarting evaluation: {total} samples")
    print(f"RoBERTa  → {ROBERTA_URL}")
    print(f"Explainer→ {LLM_EXPLAINER_URL}")
    print(f"Output   → {out_csv}\n")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for idx, item in enumerate(batch):
            code         = item.get("func", "")
            ground_truth = "vulnerable" if str(item.get("target", 0)) == "1" else "non-vulnerable"
            cwe_sampled  = item.get("cwe_sampled", "UNKNOWN")

            print(f"[{idx+1:03d}/{total}] CWE={cwe_sampled:<14} GT={ground_truth:<15} ", end="", flush=True)

            row = {f: "" for f in CSV_FIELDS}
            row["index"]        = idx + 1
            row["cwe_sampled"]  = cwe_sampled
            row["ground_truth"] = ground_truth

            t0 = time.time()
            try:
                # Step 1 — RoBERTa
                rb = call_roberta(code)
                roberta_label = rb.get("label", "non-vulnerable")
                roberta_conf  = float(rb.get("confidence", 0.0))
                row["roberta_label"]      = roberta_label
                row["roberta_confidence"] = round(roberta_conf, 4)

                # Step 2 — Dual LLM explainer
                llm = call_llm_explainer(code, roberta_label, roberta_conf)

                cl = llm.get("codellama", {})
                ds = llm.get("deepseek",  {})

                cl_agree = cl.get("llm_agreement", "unknown")   # "agrees"/"disagrees"
                ds_agree = ds.get("llm_agreement", "unknown")

                # Derive absolute verdicts from agreement relative to RoBERTa
                cl_verdict = derive_verdict(cl_agree, roberta_label)
                ds_verdict = derive_verdict(ds_agree, roberta_label)

                row["codellama_agreement"] = cl_agree
                row["deepseek_agreement"]  = ds_agree
                row["codellama_verdict"]   = cl_verdict
                row["deepseek_verdict"]    = ds_verdict

                llm_agree = (cl_verdict == ds_verdict)
                row["llm_agreement"] = llm_agree

                # Step 3 — Gemini tie-break if LLMs disagree
                if not llm_agree:
                    gemini_v = call_gemini_judge(
                        code,
                        cl.get("explanation", ""), ds.get("explanation", ""),
                        cl_verdict, ds_verdict
                    )
                    row["gemini_judge_verdict"] = gemini_v
                    final_verdict = gemini_v
                else:
                    row["gemini_judge_verdict"] = ""
                    final_verdict = cl_verdict

                row["final_verdict"] = final_verdict
                row["correct"]       = (final_verdict == ground_truth)

                # Remediation check — uses remediated_code field from your LLMResult
                has_fix = bool(
                    cl.get("remediated_code", "").strip() or
                    ds.get("remediated_code", "").strip()
                )
                row["remediation_provided"] = has_fix

                mark = "✓" if row["correct"] else "✗"
                print(f"→ {final_verdict:<15} {mark}  "
                      f"(RoBERTa={roberta_label}, CL={cl_verdict}, DS={ds_verdict})")

            except Exception as e:
                row["error"] = str(e)[:250]
                print(f"  ERROR: {e}")

            row["latency_s"] = round(time.time() - t0, 2)
            writer.writerow(row)
            f.flush()   # write incrementally — safe to Ctrl+C and resume later
            results.append(row)

            time.sleep(SLEEP_BETWEEN)

    return results


def print_summary(results: list[dict]):
    valid  = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid)

    tp = sum(1 for r in valid if r["ground_truth"]=="vulnerable"     and r["final_verdict"]=="vulnerable")
    fn = sum(1 for r in valid if r["ground_truth"]=="vulnerable"     and r["final_verdict"]=="non-vulnerable")
    fp = sum(1 for r in valid if r["ground_truth"]=="non-vulnerable" and r["final_verdict"]=="vulnerable")
    tn = sum(1 for r in valid if r["ground_truth"]=="non-vulnerable" and r["final_verdict"]=="non-vulnerable")

    tpr  = tp/(tp+fn)           if (tp+fn)>0     else 0.0
    fpr  = fp/(fp+tn)           if (fp+tn)>0     else 0.0
    prec = tp/(tp+fp)           if (tp+fp)>0     else 0.0
    f1   = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0.0
    acc  = (tp+tn)/len(valid)   if valid          else 0.0

    agree = sum(1 for r in valid if str(r.get("llm_agreement","")).lower() in ("true","1"))

    print("\n" + "="*58)
    print("  EVALUATION SUMMARY")
    print("="*58)
    print(f"  Total: {len(results)}  |  Valid: {len(valid)}  |  Errors: {errors}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy           : {acc:.3f}")
    print(f"  True Positive Rate : {tpr:.3f}  ← detection rate")
    print(f"  False Positive Rate: {fpr:.3f}  ← false alarm rate")
    print(f"  Precision          : {prec:.3f}")
    print(f"  F1 Score           : {f1:.3f}")
    print(f"  LLM Agreement      : {agree}/{len(valid)}  ({100*agree/len(valid):.1f}%)")
    print("="*58)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="test_batch.json",        help="Sampled batch JSON")
    parser.add_argument("--output", default="evaluation_results.csv", help="Results CSV")
    args = parser.parse_args()

    batch = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Loaded {len(batch)} samples from {args.input}")

    results = evaluate(batch, Path(args.output))
    print_summary(results)
    print(f"\nResults saved → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
