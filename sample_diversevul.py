"""
sample_diversevul.py
--------------------
Samples from DiverseVul (JSON-lines format) to produce a balanced test batch:
  - ~200 vulnerable functions across the top CWEs
  - 100 non-vulnerable (clean) functions

Confirmed dataset stats:
  Total vulnerable : 18,945
  Total clean      : 311,547
  CWE field        : list of strings e.g. ['CWE-787'], or [] if unknown

Usage:
    python sample_diversevul.py --input ~/Desktop/notebook/diversevul.json --output test_batch.json
"""

import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# Top CWEs confirmed from dataset, with their actual pool sizes.
# We sample 15 per CWE → 15 × 13 = 195 vulnerable + 5 from NO_CWE = 200 total.
TARGET_CWES = [
    "CWE-787",   # 2896 available
    "CWE-125",   # 1869
    "CWE-119",   # 1633
    "CWE-20",    # 1315
    "CWE-703",   # 1228
    "CWE-416",   # 1005
    "CWE-476",   # 975
    "CWE-190",   # 783
    "CWE-200",   # 747
    "CWE-399",   # 509
    "CWE-362",   # 458
    "CWE-401",   # 366
    "CWE-415",   # 269
]

VULNERABLE_PER_CWE  = 15    # 15 × 13 CWEs = 195 vulnerable
NO_CWE_EXTRA        = 5     # top up from samples with no CWE tag → total ~200
NON_VULNERABLE_COUNT = 100  # clean functions
RANDOM_SEED         = 42
# ───────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list[dict]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping malformed line {lineno}: {e}")
    return records


def sample_dataset(data: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)

    by_cwe: dict[str, list[dict]] = defaultdict(list)
    non_vulnerable: list[dict] = []

    for item in data:
        target = item.get("target", -1)
        if target == 0:
            non_vulnerable.append(item)
        elif target == 1:
            cwes = item.get("cwe", [])
            if not cwes:
                by_cwe["NO_CWE"].append(item)
            else:
                placed = False
                for cwe in cwes:
                    cwe_stripped = cwe.strip()
                    if cwe_stripped in TARGET_CWES:
                        by_cwe[cwe_stripped].append(item)
                        placed = True
                        break
                if not placed:
                    by_cwe["OTHER"].append(item)

    # Sample from each target CWE
    sampled_vulnerable: list[dict] = []
    print("\nSampling vulnerable functions by CWE:")
    for cwe in TARGET_CWES:
        pool = by_cwe[cwe]
        n = min(VULNERABLE_PER_CWE, len(pool))
        chosen = random.sample(pool, n)
        for item in chosen:
            item["_sampled_cwe"] = cwe
        sampled_vulnerable.extend(chosen)
        print(f"  {cwe:<12}  {n:>3} sampled  (pool: {len(pool)})")

    # Top up with NO_CWE samples
    no_cwe_pool = by_cwe["NO_CWE"]
    extra = random.sample(no_cwe_pool, min(NO_CWE_EXTRA, len(no_cwe_pool)))
    for item in extra:
        item["_sampled_cwe"] = "NO_CWE"
    sampled_vulnerable.extend(extra)
    print(f"  {'NO_CWE':<12}  {len(extra):>3} sampled  (pool: {len(no_cwe_pool)})")

    # Sample non-vulnerable
    sampled_clean = random.sample(non_vulnerable, min(NON_VULNERABLE_COUNT, len(non_vulnerable)))
    for item in sampled_clean:
        item["_sampled_cwe"] = "NONE"

    print(f"\nTotal vulnerable  : {len(sampled_vulnerable)}")
    print(f"Total non-vulnerable: {len(sampled_clean)}")

    # Build normalised output records
    batch = []
    for item in sampled_vulnerable + sampled_clean:
        batch.append({
            "func":        item.get("func", ""),
            "target":      item.get("target", -1),
            "cwe":         item.get("cwe", []),
            "cwe_sampled": item.get("_sampled_cwe", "UNKNOWN"),
            "project":     item.get("project", ""),
            "commit_id":   item.get("commit_id", ""),
        })

    random.shuffle(batch)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Sample DiverseVul for batch evaluation")
    parser.add_argument("--input",  default="~/Desktop/notebook/diversevul.json",
                        help="Path to DiverseVul JSONL file")
    parser.add_argument("--output", default="test_batch.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data):,} records")

    batch = sample_dataset(data)

    out = Path(args.output)
    out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(f"\nSaved {len(batch)} samples → {out.resolve()}")
    print("\nNext step:")
    print("  python run_batch_evaluation.py --input test_batch.json --output evaluation_results.csv")


if __name__ == "__main__":
    main()
