"""
Pipeline Evaluation Script — DiverseVul Batch Runner
-----------------------------------------------------
Loads 50 balanced samples from DiverseVul (25 vulnerable + 25 non-vulnerable),
runs each through the full pipeline:
    1. RoBERTa API (port 8000) → prediction + confidence
    2. LLM Explainer API (port 8001) → confirmation + explanation + remediation

Saves results to: evaluation_results.csv

Usage:
    python run_evaluation.py --dataset /path/to/diversevul_20230702.json

Example (OneDrive path on Mac):
    python run_evaluation.py \
      --dataset ~/Library/CloudStorage/OneDrive-Personal/diversevul_20230702.json
"""

import argparse
import json
import time
import requests
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ROBERTA_URL = "http://127.0.0.1:8000/predict"
EXPLAINER_URL = "http://127.0.0.1:8001/explain"
N_SAMPLES = 25          # per class → 50 total
RANDOM_SEED = 42
OUTPUT_FILE = "evaluation_results.csv"
REQUEST_TIMEOUT = 180   # seconds — generous for CodeLlama

# ── Argument parser ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run DiverseVul evaluation pipeline")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to diversevul_20230702.json"
)
parser.add_argument(
    "--skip-llm",
    action="store_true",
    help="Skip LLM explanation step (faster, ML predictions only)"
)
args = parser.parse_args()

# ── Load dataset ───────────────────────────────────────────────────────────────

dataset_path = Path(args.dataset).expanduser()
if not dataset_path.exists():
    print(f"❌ Dataset not found at: {dataset_path}")
    print("   Check the path and try again.")
    exit(1)

print(f"📂 Loading dataset from: {dataset_path}")
data = []
with open(dataset_path, 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} total samples")
print(f"   Vulnerable:     {df['target'].sum()}")
print(f"   Non-vulnerable: {(df['target'] == 0).sum()}")

# ── Sample balanced dataset ────────────────────────────────────────────────────

vulnerable_df = df[df['target'] == 1].sample(n=N_SAMPLES, random_state=RANDOM_SEED)
non_vulnerable_df = df[df['target'] == 0].sample(n=N_SAMPLES, random_state=RANDOM_SEED)
sample_df = pd.concat([vulnerable_df, non_vulnerable_df]).sample(
    frac=1, random_state=RANDOM_SEED  # shuffle
).reset_index(drop=True)

print(f"\n🎯 Sampled {len(sample_df)} balanced examples")
print(f"   25 vulnerable + 25 non-vulnerable\n")

# ── Check services are running ─────────────────────────────────────────────────

print("🔍 Checking services...")
for name, url in [("RoBERTa API", "http://127.0.0.1:8000"), ("LLM Explainer", "http://127.0.0.1:8001")]:
    try:
        r = requests.get(url, timeout=5)
        print(f"   ✅ {name} is running")
    except Exception:
        print(f"   ❌ {name} is NOT running at {url}")
        print(f"      Start it before running this script.")
        exit(1)

print()

# ── Run pipeline ───────────────────────────────────────────────────────────────

results = []

for i, row in sample_df.iterrows():
    idx = results.__len__() + 1
    code = str(row['func'])
    true_label = int(row['target'])
    cwe = row.get('cwe', [])
    cwe_str = ", ".join(cwe) if isinstance(cwe, list) else str(cwe)

    print(f"[{idx:02d}/50] True label: {'VULNERABLE' if true_label == 1 else 'NON-VULNERABLE'} | CWE: {cwe_str[:50]}")

    result = {
        "sample_id": idx,
        "true_label": true_label,
        "true_label_str": "vulnerable" if true_label == 1 else "non-vulnerable",
        "cwe": cwe_str,
        "func_preview": code[:100].replace('\n', ' '),
        # RoBERTa fields
        "ml_predicted_label": None,
        "ml_predicted_label_id": None,
        "ml_confidence": None,
        "ml_prob_vulnerable": None,
        "ml_prob_non_vulnerable": None,
        "ml_correct": None,
        # LLM fields
        "llm_agreement": None,
        "llm_explanation": None,
        "llm_remediation": None,
        "error": None
    }

    # Step 1: RoBERTa prediction
    try:
        ml_response = requests.post(
            ROBERTA_URL,
            json={"code": code},
            timeout=30
        )
        ml_response.raise_for_status()
        ml_result = ml_response.json()

        result["ml_predicted_label"] = ml_result["label"]
        result["ml_predicted_label_id"] = ml_result["label_id"]
        result["ml_confidence"] = ml_result["confidence"]
        result["ml_prob_vulnerable"] = ml_result["probabilities"]["vulnerable"]
        result["ml_prob_non_vulnerable"] = ml_result["probabilities"]["non-vulnerable"]
        result["ml_correct"] = (ml_result["label_id"] == true_label)

        print(f"         ML → {ml_result['label']} ({ml_result['confidence']:.2%}) | Correct: {result['ml_correct']}")

    except Exception as e:
        result["error"] = f"RoBERTa error: {str(e)}"
        print(f"         ❌ RoBERTa error: {e}")
        results.append(result)
        continue

    # Step 2: LLM explanation (skip if --skip-llm flag is set)
    if not args.skip_llm:
        try:
            llm_response = requests.post(
                EXPLAINER_URL,
                json={
                    "code": code,
                    "predicted_label": ml_result["label"],
                    "confidence": ml_result["confidence"]
                },
                timeout=REQUEST_TIMEOUT
            )
            llm_response.raise_for_status()
            llm_result = llm_response.json()

            result["llm_agreement"] = llm_result["llm_agreement"]
            result["llm_explanation"] = llm_result["explanation"]
            result["llm_remediation"] = llm_result["remediation"]

            print(f"         LLM → {llm_result['llm_agreement'].upper()}")

        except Exception as e:
            result["error"] = f"LLM error: {str(e)}"
            print(f"         ⚠️  LLM error: {e}")

    results.append(result)

    # Small pause between requests to avoid overwhelming the services
    time.sleep(1)

# ── Save results ───────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Results saved to: {OUTPUT_FILE}")

# ── Print summary statistics ───────────────────────────────────────────────────

print("\n" + "="*50)
print("📊 EVALUATION SUMMARY")
print("="*50)

total = len(results_df)
ml_correct = results_df['ml_correct'].sum()
ml_accuracy = ml_correct / total * 100

print(f"\nTotal samples:     {total}")
print(f"ML correct:        {ml_correct}/{total} ({ml_accuracy:.1f}%)")

# Breakdown by true label
for label_val, label_name in [(1, "Vulnerable"), (0, "Non-vulnerable")]:
    subset = results_df[results_df['true_label'] == label_val]
    correct = subset['ml_correct'].sum()
    print(f"  {label_name}: {correct}/{len(subset)} correct")

# LLM agreement stats
if not args.skip_llm:
    agrees = (results_df['llm_agreement'] == 'agrees').sum()
    disagrees = (results_df['llm_agreement'] == 'disagrees').sum()
    print(f"\nLLM agrees with ML:    {agrees}/{total} ({agrees/total*100:.1f}%)")
    print(f"LLM disagrees with ML: {disagrees}/{total} ({disagrees/total*100:.1f}%)")

    # Disagreements where ML was wrong
    ml_wrong_llm_disagrees = results_df[
        (results_df['ml_correct'] == False) &
        (results_df['llm_agreement'] == 'disagrees')
    ]
    print(f"\nML wrong + LLM disagrees (LLM caught ML errors): {len(ml_wrong_llm_disagrees)}")

    ml_wrong_llm_agrees = results_df[
        (results_df['ml_correct'] == False) &
        (results_df['llm_agreement'] == 'agrees')
    ]
    print(f"ML wrong + LLM agrees (both wrong):              {len(ml_wrong_llm_agrees)}")

print("\n✅ Evaluation complete!")
print(f"   Open {OUTPUT_FILE} for full results.")
