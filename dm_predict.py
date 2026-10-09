"""Готовит входы для digital-metrics.

GT (один раз):
    python -m fcos_train.dm_predict gt --out-dir dm \
        --val-ann  .../annotations/instances_val.json  --val-img-dir  .../val \
        --test-ann .../annotations/instances_test.json --test-img-dir .../test \
        [--train-ann .../annotations/instances_train.json]     # только для подсчёта примеров train

Предсказания (на каждую модель, бэкенды и флаги — как в bench_final):
    python -m fcos_train.dm_predict preds --name frcnn_torch --backend torch --config ... --ckpt ... \
        --val-ann ... --val-img-dir ... --test-ann ... --test-img-dir ... --out-dir dm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from fcos_train.bench_final import check_args, make_runner, read_rgb
except ImportError:                      # bench_final.py лежит рядом (окружения YOLO / mmdet)
    from bench_final import check_args, make_runner, read_rgb

BOX = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def image_key(split: str, file_name: str) -> str:
    """Уникальное имя кадра: один кадр — один сплит."""
    return f"{split}/{file_name}"


def load_coco(ann_file: str):
    with open(ann_file) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    return coco, names


def splits_from_args(a, with_train: bool):
    out = []
    if with_train and a.train_ann:
        out.append(("train", a.train_ann, a.train_img_dir))
    out.append(("val", a.val_ann, a.val_img_dir))
    out.append(("test", a.test_ann, a.test_img_dir))
    return out


# ------------------------------------------------------------------ GT
def build_gt(a) -> pd.DataFrame:
    rows, skipped, ref_names = [], 0, None
    for split, ann_file, img_dir in splits_from_args(a, with_train=True):
        coco, names = load_coco(ann_file)
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            print(f"[warn] категории в {ann_file} отличаются от первого сплита: {names} vs {ref_names}")

        by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            by_img.setdefault(ann["image_id"], []).append((names[ann["category_id"]], x, y, x + w, y + h))

        for im in coco["images"]:
            base = {"image_name": image_key(split, im["file_name"]), "split": split,
                    "image_path": os.path.join(img_dir, im["file_name"]) if img_dir else None,
                    "image_width": im.get("width"), "image_height": im.get("height")}
            anns = by_img.get(im["id"])
            if not anns:                                     # пустой кадр — строка-заглушка
                rows.append({**base, "instance_label": None, **{k: np.nan for k in BOX}})
                continue
            for label, x1, y1, x2, y2 in anns:
                rows.append({**base, "instance_label": label,
                             "bbox_x_tl": x1, "bbox_y_tl": y1, "bbox_x_br": x2, "bbox_y_br": y2})

    gt = pd.DataFrame(rows)
    if skipped:
        print(f"[warn] пропущено {skipped} GT-боксов нулевой площади")
    summary = gt[gt["instance_label"].notna()].groupby(["split", "instance_label"]).size().unstack(0)
    print("GT, объектов по сплитам:\n", summary.fillna(0).astype(int).to_string())
    print("кадров по сплитам:", gt.groupby("split")["image_name"].nunique().to_dict())
    return gt


# ------------------------------------------------------------------ predictions
def build_preds(a) -> pd.DataFrame:
    runner_args = SimpleNamespace(
        backend=a.backend, export_dir=a.export_dir, ep=a.ep, config=a.config, ckpt=a.ckpt, amp=a.amp,
        weights=a.weights, imgsz=a.imgsz, half=a.half, conf=a.conf, iou=a.iou,
        model_dir=a.model_dir, classes=a.classes, ann_file=a.val_ann, device_id=a.device_id,
    )
    check_args(runner_args)
    runner, info = make_runner(runner_args)
    print("модель:", a.name, "|", info)

    rows, unknown, degenerate, n_img = [], 0, 0, 0
    t0 = time.perf_counter()
    for split, ann_file, img_dir in splits_from_args(a, with_train=False):
        coco, names = load_coco(ann_file)
        for im in coco["images"]:
            r = runner(read_rgb(os.path.join(img_dir, im["file_name"])))
            key = image_key(split, im["file_name"])
            for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
                label = names.get(int(c))
                if label is None:
                    unknown += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in b)
                if not (x2 > x1 and y2 > y1):
                    degenerate += 1
                    continue
                rows.append({"image_name": key, "instance_label": label,
                             "bbox_x_tl": x1, "bbox_y_tl": y1, "bbox_x_br": x2, "bbox_y_br": y2,
                             "confidence": float(min(max(s, 0.0), 1.0))})
            n_img += 1
            if n_img % 500 == 0:
                print(f"  {n_img} кадров, {time.perf_counter() - t0:.0f} с")

    if unknown:
        print(f"[warn] {unknown} детекций с неизвестным классом пропущены — проверьте маппинг классов")
    if degenerate:
        print(f"[warn] {degenerate} боксов нулевой площади пропущены")
    print(f"готово: {n_img} кадров, {len(rows)} детекций")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gt", "preds"])
    ap.add_argument("--out-dir", default="dm")
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--val-img-dir")
    ap.add_argument("--test-ann", required=True)
    ap.add_argument("--test-img-dir")
    ap.add_argument("--train-ann")
    ap.add_argument("--train-img-dir")
    # модель (как в bench_final)
    ap.add_argument("--name", help="preds: имя модели, пойдёт в имя файла и в сводку")
    ap.add_argument("--backend", choices=["onnx", "onnx1", "torch", "yolo", "mmdet", "mmdeploy"])
    ap.add_argument("--export-dir")
    ap.add_argument("--ep", choices=["cuda", "trt", "trt_fp16"], default="cuda")
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--model-dir")
    ap.add_argument("--classes")
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    if a.what == "gt":
        gt = build_gt(a)
        path = os.path.join(a.out_dir, "gt.csv")
        gt.to_csv(path)
    else:
        if not (a.name and a.backend and a.val_img_dir and a.test_img_dir):
            raise SystemExit("preds требует --name, --backend, --val-img-dir, --test-img-dir")
        preds = build_preds(a)
        path = os.path.join(a.out_dir, f"preds_{a.name}.csv")
        preds.to_csv(path)
    print("сохранено:", path)


if __name__ == "__main__":
    main()
