"""Прогоняет все preds_*.csv через digital-metrics: порог по классам на val, отчёт на test.

    python -m fcos_train.dm_evaluate --gt dm/gt.csv --preds dm/preds_*.csv --out-dir dm/results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from digital_metrics import Evaluation


def model_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("preds_"):] if stem.startswith("preds_") else stem


def run_one(gt: pd.DataFrame, preds_path: str, a) -> dict:
    name = model_name(preds_path)
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)

    preds = pd.read_csv(preds_path, index_col=0)
    ev = Evaluation(preds, gt.copy(), iou_threshold=a.iou, skip_cohen_kappa=True)
    ev(split=a.split, calibration_split=a.calibration_split)

    rows = []
    for cls, m in ev.metrics.items():
        rows.append({"class": cls, "tp": m.tp, "fp": m.fp, "fn": m.fn, "confidence": m.confidence,
                     "precision": m.precision, "recall": m.recall, "f1_score": m.f1_score,
                     "ap50": m.ap50, "ap75": m.ap75, "ap50_95": m.ap50_95})
    per_class = pd.DataFrame(rows).set_index("class")
    per_class.round(4).to_csv(os.path.join(out, "metrics.csv"))
    with open(os.path.join(out, "best_confidences.json"), "w") as f:
        json.dump(ev.best_confidences, f, indent=2, ensure_ascii=False)
    ev.get_dashboards(save_to_excel=True, path=out)

    print(f"\n=== {name} ===\n{per_class.round(4).to_string()}")
    return {"model": name,
            "mAP50": np.nanmean(per_class["ap50"]),
            "mAP50_95": np.nanmean(per_class["ap50_95"]),
            "mean_precision": per_class["precision"].mean(),
            "mean_recall": per_class["recall"].mean(),
            "mean_f1": per_class["f1_score"].mean(),
            "tp": int(per_class["tp"].sum()), "fp": int(per_class["fp"].sum()),
            "fn": int(per_class["fn"].sum()),
            **{f"ap50_{c}": v for c, v in per_class["ap50"].items()},
            **{f"f1_{c}": v for c, v in per_class["f1_score"].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out-dir", default="dm/results")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibration-split", default="val")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    gt = pd.read_csv(a.gt, index_col=0)
    summary = pd.DataFrame([run_one(gt, p, a) for p in a.preds]).set_index("model")
    summary = summary.sort_values("mAP50_95", ascending=False)
    summary.round(4).to_csv(os.path.join(a.out_dir, "summary.csv"))
    print("\n=== СВОДКА ===\n", summary.round(4).to_string())


if __name__ == "__main__":
    main()
