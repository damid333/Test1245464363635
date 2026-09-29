python -m fcos_train.bench_stages \
    --graphs runs/train2/two_graphs \
    --config runs/train2/config.yaml \
    --ckpt runs/train2/best.pth \
    --images /mnt/disk01/data/di.lashenkov/dataset/nta4/coco_20260728_2class/val

# src/fcos_train/bench_stages.py
from __future__ import annotations

import argparse
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection.rpn import concat_box_prediction_layers

from .config import load_config
from .inference_v2 import TwoGraphFrcnn, load_image
from .model import build_model


def _t():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def run_with_timing(m: TwoGraphFrcnn, img: torch.Tensor, acc: dict):
    t0 = _t()
    orig = [(int(img.shape[-2]), int(img.shape[-1]))]
    il, _ = m.transform([img], None)
    il.tensors = m._pad_to_export(il.tensors)
    x = il.tensors
    t1 = _t(); acc["1. transform"].append(t1 - t0)

    outs = m._run(m.sess1, {"images": x})
    n = m.n_lvl
    feats, obj_l, dlt_l = outs[:n], outs[n:2 * n], outs[2 * n:3 * n]
    t2 = _t(); acc["2. граф1 backbone+FPN+RPN (onnx)"].append(t2 - t1)

    key = tuple(x.shape)
    if key not in m._anchor_cache:
        m._anchor_cache[key] = m.rpn.anchor_generator(il, feats)
    anchors = m._anchor_cache[key]
    napl = [o[0].numel() for o in obj_l]
    obj, dlt = concat_box_prediction_layers(obj_l, dlt_l)
    props = m.rpn.box_coder.decode(dlt, anchors).view(len(anchors), -1, 4)
    props, _ = m.rpn.filter_proposals(props, obj, il.image_sizes, napl)
    t3 = _t(); acc["3. RPN decode+NMS (torch)"].append(t3 - t2)

    pooled = m.roi_heads.box_roi_pool(
        OrderedDict(zip(m.fpn_keys, feats)), props, il.image_sizes)
    t4 = _t(); acc["4. RoIAlign (torch)"].append(t4 - t3)
    acc["   число областей"].append(pooled.shape[0])

    cls, reg = m._run(m.sess2, {"pooled": pooled})
    t5 = _t(); acc["5. граф2 классификатор (onnx)"].append(t5 - t4)

    b, s, l = m.roi_heads.postprocess_detections(cls.float(), reg.float(), props, il.image_sizes)
    m.transform.postprocess([{"boxes": b[0], "scores": s[0], "labels": l[0]}],
                            il.image_sizes, orig)
    t6 = _t(); acc["6. decode+NMS финал (torch)"].append(t6 - t5)
    acc["ИТОГО кадр"].append(t6 - t0)


def bench_classifier(m: TwoGraphFrcnn, torch_model, n_iter=200):
    """Классификатор изолированно: ONNX vs torch на одном и том же входе."""
    C, P = m.meta["out_channels"], m.meta["pool_size"]
    head_t = torch.nn.Sequential(torch_model.roi_heads.box_head).eval()
    pred_t = torch_model.roi_heads.box_predictor.eval()

    print("\nКлассификатор изолированно (мс на вызов, median):")
    print(f"  {'областей':>9} | {'onnx':>8} | {'torch':>8}")
    for n_rois in (100, 300, 1000):
        x = torch.randn(n_rois, C, P, P, device=m.device)
        res = {}
        for name, fn in [
            ("onnx",  lambda: m._run(m.sess2, {"pooled": x})),
            ("torch", lambda: pred_t(head_t(x))),
        ]:
            with torch.no_grad():
                for _ in range(20):
                    fn()
                ts = []
                for _ in range(n_iter):
                    t0 = _t(); fn(); ts.append((_t() - t0) * 1000)
            res[name] = np.median(ts)
        print(f"  {n_rois:>9} | {res['onnx']:8.2f} | {res['torch']:8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--fp16", action="store_true")
    a = p.parse_args()

    m = TwoGraphFrcnn(a.graphs, fp16=a.fp16)

    paths = sorted(list(Path(a.images).glob("*.jpg")) + list(Path(a.images).glob("*.png")))[:10]
    imgs = [load_image(p, m.device) for p in paths]

    warm = defaultdict(list)
    for i in range(20):
        run_with_timing(m, imgs[i % len(imgs)], warm)

    acc = defaultdict(list)
    for i in range(a.n):
        run_with_timing(m, imgs[i % len(imgs)], acc)

    print(f"\nРазбивка кадра по этапам (n={a.n}, median):")
    total = np.median(acc["ИТОГО кадр"]) * 1000
    for k, v in acc.items():
        if k.startswith("   "):
            print(f"  {k:38s} {np.median(v):8.0f}")
        else:
            ms = np.median(v) * 1000
            print(f"  {k:38s} {ms:8.2f} мс  ({ms / total * 100:4.1f}%)")

    # torch-голова с весами из чекпойнта, для сравнения
    cfg = load_config(a.config)
    ck = torch.load(a.ckpt, map_location="cpu")
    tm = build_model(cfg.model, int(ck.get("num_classes", cfg.model.num_classes)))
    tm.load_state_dict(ck["model"] if "model" in ck else ck)
    tm.eval().to(m.device)
    bench_classifier(m, tm)
