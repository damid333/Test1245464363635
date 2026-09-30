"""Сверка детекций ONNX (два графа) с исходной PyTorch-моделью на реальных картинках.

    python check_parity.py --config runs/exp_X/config.yaml --ckpt runs/exp_X/best.pth \
        --export-dir export/exp_X --img-dir data/val/images --ann-file data/val/ann.json --n 50
"""
from __future__ import annotations

import argparse

import numpy as np

from bench import TorchRunner, load_val
from onnx_detector import OnnxDetector, read_rgb


def iou_matrix(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def match(ref, got, score_min):
    """Жадно сопоставляет боксы с одинаковым классом. Возвращает IoU и |Δscore| пар + число несопоставленных."""
    r = ref["scores"] >= score_min
    g = got["scores"] >= score_min
    rb, rs, rl = ref["boxes"][r], ref["scores"][r], ref["labels"][r]
    gb, gs, gl = got["boxes"][g], got["scores"][g], got["labels"][g]
    if len(rb) == 0 or len(gb) == 0:
        return [], [], len(rb) + len(gb)
    iou = iou_matrix(rb, gb) * (rl[:, None] == gl[None, :])
    used, ious, dscore = set(), [], []
    for i in np.argsort(-rs):
        row = iou[i].copy()
        row[list(used)] = -1.0
        j = int(np.argmax(row))
        if row[j] > 0.5:
            used.add(j)
            ious.append(iou[i, j])
            dscore.append(abs(rs[i] - gs[j]))
    return ious, dscore, (len(rb) - len(ious)) + (len(gb) - len(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--score-min", type=float, default=0.3, help="сравниваем только уверенные детекции")
    a = ap.parse_args()

    torch_r = TorchRunner(a.config, a.ckpt)
    onnx_r = OnnxDetector(a.export_dir)
    print("onnx providers:", onnx_r.providers)

    all_iou, all_ds, unmatched, n_ref = [], [], 0, 0
    for _, path in load_val(a.ann_file, a.img_dir, a.n):
        img = read_rgb(path)
        ref, got = torch_r(img), onnx_r(img)
        ious, ds, um = match(ref, got, a.score_min)
        all_iou += ious
        all_ds += ds
        unmatched += um
        n_ref += int((ref["scores"] >= a.score_min).sum())

    print(f"\nкартинок: {a.n}, детекций torch (score>={a.score_min}): {n_ref}")
    print(f"сопоставлено: {len(all_iou)}, несопоставлено (обе стороны): {unmatched}")
    if all_iou:
        iou = np.asarray(all_iou)
        print(f"IoU: mean={iou.mean():.4f} min={iou.min():.4f} доля>=0.99: {(iou >= 0.99).mean():.3f}")
        print(f"|Δscore|: mean={np.mean(all_ds):.2e} max={np.max(all_ds):.2e}")
    ok = n_ref == 0 or (unmatched <= 0.02 * n_ref and (not all_iou or np.mean(all_iou) >= 0.98))
    print("ИТОГ:", "OK" if ok else "РАСХОЖДЕНИЕ — смотреть preprocessing/canvas/fp16")


if __name__ == "__main__":
    main()
