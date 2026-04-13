"""
compute_metrics.py
------------------
Reads evaluation_results.csv and produces:
  1. Console summary (TPR, FPR, Precision, F1, agreement rate)
  2. Per-CWE breakdown table (saved as per_cwe_metrics.csv)
  3. Confusion matrix + per-CWE bar chart (saved as evaluation_charts.png)

Usage:
    python compute_metrics.py --input evaluation_results.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠ matplotlib not installed — charts will be skipped. Run: pip install matplotlib")


# ── Load CSV ───────────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_bool(val) -> bool:
    return str(val).lower() in ("true", "1", "yes")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("error")]
    tp = sum(1 for r in valid if r["ground_truth"] == "vulnerable"     and r["final_verdict"] == "vulnerable")
    fn = sum(1 for r in valid if r["ground_truth"] == "vulnerable"     and r["final_verdict"] == "non-vulnerable")
    fp = sum(1 for r in valid if r["ground_truth"] == "non-vulnerable" and r["final_verdict"] == "vulnerable")
    tn = sum(1 for r in valid if r["ground_truth"] == "non-vulnerable" and r["final_verdict"] == "non-vulnerable")

    tpr  = tp / (tp + fn)   if (tp + fn)   > 0 else 0.0
    fpr  = fp / (fp + tn)   if (fp + tn)   > 0 else 0.0
    prec = tp / (tp + fp)   if (tp + fp)   > 0 else 0.0
    f1   = 2*prec*tpr / (prec+tpr) if (prec+tpr) > 0 else 0.0
    acc  = (tp+tn) / len(valid) if valid else 0.0

    agree = sum(1 for r in valid if to_bool(r.get("llm_agreement", False)))
    disagree = len(valid) - agree

    rb_correct  = sum(1 for r in valid if r.get("roberta_label") == r["ground_truth"])
    rem_provided = sum(1 for r in valid if to_bool(r.get("remediation_provided")))
    rem_verified = sum(1 for r in valid if to_bool(r.get("remediation_verified")))

    return dict(
        total=len(rows), valid=len(valid), errors=len(rows)-len(valid),
        tp=tp, fn=fn, fp=fp, tn=tn,
        tpr=tpr, fpr=fpr, precision=prec, f1=f1, accuracy=acc,
        llm_agree=agree, llm_disagree=disagree,
        roberta_correct=rb_correct,
        remediation_provided=rem_provided,
        remediation_verified=rem_verified,
    )


def compute_per_cwe(rows: list[dict]) -> list[dict]:
    by_cwe: dict[str, list] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_cwe[r.get("cwe_sampled", "UNKNOWN")].append(r)

    per_cwe = []
    for cwe, items in sorted(by_cwe.items()):
        tp = sum(1 for r in items if r["ground_truth"]=="vulnerable"     and r["final_verdict"]=="vulnerable")
        fn = sum(1 for r in items if r["ground_truth"]=="vulnerable"     and r["final_verdict"]=="non-vulnerable")
        fp = sum(1 for r in items if r["ground_truth"]=="non-vulnerable" and r["final_verdict"]=="vulnerable")
        tn = sum(1 for r in items if r["ground_truth"]=="non-vulnerable" and r["final_verdict"]=="non-vulnerable")
        tpr  = tp/(tp+fn)  if (tp+fn)>0  else 0.0
        fpr  = fp/(fp+tn)  if (fp+tn)>0  else 0.0
        prec = tp/(tp+fp)  if (tp+fp)>0  else 0.0
        f1   = 2*prec*tpr/(prec+tpr) if (prec+tpr)>0 else 0.0
        per_cwe.append(dict(
            cwe=cwe, n=len(items),
            tp=tp, fn=fn, fp=fp, tn=tn,
            tpr=round(tpr,3), fpr=round(fpr,3),
            precision=round(prec,3), f1=round(f1,3),
        ))
    return per_cwe


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary(m: dict):
    print("\n" + "="*58)
    print("  OVERALL EVALUATION METRICS")
    print("="*58)
    print(f"  Samples evaluated    : {m['valid']}  (errors: {m['errors']})")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  Accuracy             : {m['accuracy']:.3f}")
    print(f"  True Positive Rate   : {m['tpr']:.3f}  ← detection capability")
    print(f"  False Positive Rate  : {m['fpr']:.3f}  ← false alarm rate")
    print(f"  Precision            : {m['precision']:.3f}")
    print(f"  F1 Score             : {m['f1']:.3f}")
    print(f"  RoBERTa accuracy     : {m['roberta_correct']}/{m['valid']}"
          f"  ({100*m['roberta_correct']/m['valid']:.1f}%)")
    print(f"  LLM agreement        : {m['llm_agree']}/{m['valid']}"
          f"  ({100*m['llm_agree']/m['valid']:.1f}%)")
    print(f"  Remediation provided : {m['remediation_provided']}/{m['valid']}")
    print("="*58)


def print_per_cwe(per_cwe: list[dict]):
    print("\n  PER-CWE BREAKDOWN")
    print(f"  {'CWE':<14} {'N':>4} {'TPR':>6} {'FPR':>6} {'Prec':>6} {'F1':>6}")
    print("  " + "-"*46)
    for row in per_cwe:
        print(f"  {row['cwe']:<14} {row['n']:>4} {row['tpr']:>6.3f} {row['fpr']:>6.3f}"
              f" {row['precision']:>6.3f} {row['f1']:>6.3f}")


def save_per_cwe_csv(per_cwe: list[dict], out_path: str):
    fields = ["cwe", "n", "tp", "fn", "fp", "tn", "tpr", "fpr", "precision", "f1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_cwe)
    print(f"\n  Per-CWE CSV saved → {out_path}")


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot_charts(m: dict, per_cwe: list[dict], out_path: str):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Vulnerability Detection Pipeline — Evaluation Results", fontsize=14, fontweight="bold")

    # Chart 1: Confusion matrix heatmap
    ax = axes[0]
    matrix = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Clean", "Pred: Vuln"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: Clean", "Actual: Vuln"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if matrix[i][j] > (m["valid"]/4) else "black")

    # Chart 2: Overall metrics bar
    ax = axes[1]
    labels = ["Accuracy", "TPR\n(Detection)", "FPR\n(False Alarm)", "Precision", "F1 Score"]
    values = [m["accuracy"], m["tpr"], m["fpr"], m["precision"], m["f1"]]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall Metrics")
    ax.set_ylabel("Score")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Chart 3: Per-CWE TPR bar chart
    ax = axes[2]
    cwes = [r["cwe"] for r in per_cwe if r["cwe"] != "NONE"]
    tprs = [r["tpr"] for r in per_cwe if r["cwe"] != "NONE"]
    y_pos = range(len(cwes))
    bar_colors = ["#55A868" if t >= 0.7 else "#C44E52" if t < 0.4 else "#CCB974" for t in tprs]
    ax.barh(list(y_pos), tprs, color=bar_colors, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cwes, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("True Positive Rate")
    ax.set_title("Detection Rate by CWE")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    green_patch = mpatches.Patch(color="#55A868", label="≥ 0.70")
    yellow_patch = mpatches.Patch(color="#CCB974", label="0.40–0.69")
    red_patch = mpatches.Patch(color="#C44E52", label="< 0.40")
    ax.legend(handles=[green_patch, yellow_patch, red_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Charts saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="evaluation_results.csv")
    parser.add_argument("--cwe_csv", default="per_cwe_metrics.csv")
    parser.add_argument("--chart",  default="evaluation_charts.png")
    args = parser.parse_args()

    rows = load_results(args.input)
    m = compute_metrics(rows)
    per_cwe = compute_per_cwe(rows)

    print_summary(m)
    print_per_cwe(per_cwe)
    save_per_cwe_csv(per_cwe, args.cwe_csv)
    plot_charts(m, per_cwe, args.chart)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
