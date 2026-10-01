"""Сверка детекций ONNX с исходной PyTorch-моделью на реальных картинках.

Два графа:
    python -m fcos_train.check_parity --config .../config.yaml --ckpt .../best.pth \
        --export-dir .../onnx --img-dir .../val --ann-file .../instances_val.json --n 50
Один граф:
    ... --single --export-dir .../onnx_single
"""
from __future__ import annotations

import argparse

import numpy as np

from fcos_train.bench_final import TorchRunner, load_val
from fcos_train.inference_onnx_final import OnnxDetector, OnnxSingleDetector, read_rgb


def iou_matrix(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def match(ref, got, score_min):
    """Жадное сопоставление боксов одного класса.
    Возвращает IoU пар, |Δscore| пар, число детекций ref и got выше порога."""
    r = ref["scores"] >= score_min
    g = got["scores"] >= score_min
    rb, rs, rl = ref["boxes"][r], ref["scores"][r], ref["labels"][r]
    gb, gs, gl = got["boxes"][g], got["scores"][g], got["labels"][g]
    ious, dscore = [], []
    if len(rb) and len(gb):
        iou = iou_matrix(rb, gb) * (rl[:, None] == gl[None, :])
        used = set()
        for i in np.argsort(-rs):
            row = iou[i].copy()
            if used:
                row[list(used)] = -1.0
            j = int(np.argmax(row))
            if row[j] > 0.5:
                used.add(j)
                ious.append(float(iou[i, j]))
                dscore.append(float(abs(rs[i] - gs[j])))
    return ious, dscore, len(rb), len(gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--single", action="store_true", help="классический одиночный граф (model.onnx)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--score-min", type=float, default=0.3, help="сравниваем только уверенные детекции")
    ap.add_argument("--show", type=int, default=5, help="сколько худших кадров вывести подробно")
    a = ap.parse_args()

    torch_r = TorchRunner(a.config, a.ckpt)
    onnx_r = OnnxSingleDetector(a.export_dir) if a.single else OnnxDetector(a.export_dir)
    print("режим:", "один граф" if a.single else "два графа", "| onnx providers:", onnx_r.providers)

    all_iou, all_ds = [], []
    n_ref = n_got = 0
    per_image = []
    for _, path in load_val(a.ann_file, a.img_dir, a.n):
        img = read_rgb(path)
        ref, got = torch_r(img), onnx_r(img)
        ious, ds, nr, ng = match(ref, got, a.score_min)
        all_iou += ious
        all_ds += ds
        n_ref += nr
        n_got += ng
        per_image.append((nr + ng - 2 * len(ious), nr, ng, len(ious), path))

    matched = len(all_iou)
    unmatched_ref, unmatched_got = n_ref - matched, n_got - matched
    print(f"\nкартинок: {a.n}, порог score >= {a.score_min}")
    print(f"детекций: torch={n_ref}  onnx={n_got}")
    print(f"сопоставлено: {matched}  | только в torch: {unmatched_ref}  | только в onnx: {unmatched_got}")
    if all_iou:
        iou = np.asarray(all_iou)
        print(f"IoU пар: mean={iou.mean():.4f} min={iou.min():.4f} доля>=0.99: {(iou >= 0.99).mean():.3f}")
        print(f"|Δscore|: mean={np.mean(all_ds):.2e} max={np.max(all_ds):.2e}")

    worst = sorted(per_image, key=lambda t: -t[0])[: a.show]
    if worst and worst[0][0] > 0:
        print(f"\nхудшие кадры (несопоставлено / torch / onnx / пар):")
        for um, nr, ng, m, p in worst:
            if um:
                print(f"  {um:3d} / {nr:3d} / {ng:3d} / {m:3d}  {p}")

    ok = n_ref == 0 or (
        (unmatched_ref + unmatched_got) <= 0.02 * n_ref and (not all_iou or np.mean(all_iou) >= 0.98)
    )
    print("\nИТОГ:", "OK" if ok else "РАСХОЖДЕНИЕ")


if __name__ == "__main__":
    main()
