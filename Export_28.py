# src/fcos_train/export_v2.py
"""Экспорт Faster R-CNN в два ONNX-графа.

Граф 1: backbone + FPN + RPN head       (только свёртки, всё на CUDA EP)
Граф 2: box_head + box_predictor        (conv/FC, всё на CUDA EP)
Постобработка (anchors, decode, NMS, RoIAlign) выполняется в torchvision
на GPU во время инференса, весов в ней нет.

Запуск:
    python -m fcos_train.export_v2 \
        --config runs/train2/config.yaml \
        --ckpt   runs/train2/best.pth \
        --out    runs/train2/two_graphs \
        [--fp16] [--static --height 160 --width 768]
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

from .config import load_config
from .model import build_model


class Graph1_BackboneRPNHead(nn.Module):
    """backbone + FPN + RPN head. Без topk/NMS, поэтому без Memcpy."""

    def __init__(self, model, fpn_keys):
        super().__init__()
        self.backbone = model.backbone
        self.rpn_head = model.rpn.head
        self.fpn_keys = fpn_keys

    def forward(self, images: torch.Tensor):
        feats = self.backbone(images)
        fl = [feats[k] for k in self.fpn_keys]
        objectness, deltas = self.rpn_head(fl)
        return tuple(fl) + tuple(objectness) + tuple(deltas)


class Graph2_BoxHead(nn.Module):
    """box_head + box_predictor -> cls_logits, box_regression."""

    def __init__(self, model):
        super().__init__()
        self.box_head = model.roi_heads.box_head
        self.box_predictor = model.roi_heads.box_predictor

    def forward(self, pooled: torch.Tensor):
        feat = self.box_head(pooled)
        cls_logits, box_reg = self.box_predictor(feat)
        return cls_logits, box_reg


def _load_model(cfg_path: str, ckpt_path: str):
    cfg = load_config(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    if cfg.model.arch != "faster_rcnn":
        raise ValueError(f"экспортёр для arch: faster_rcnn, получено {cfg.model.arch}")

    model = build_model(cfg.model, num_classes)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().cuda()
    return cfg, model, num_classes


def _to_fp16(path: str) -> str:
    import onnx
    from onnxconverter_common import float16

    m = onnx.load(path)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    out = path.replace(".onnx", "_fp16.onnx")
    onnx.save(m16, out)
    print(f"[ok] {out}  ({os.path.getsize(out) / 2**20:.1f} MB)")
    return out


def export(cfg_path, ckpt_path, out_dir, height=160, width=768,
           opset=16, static=False, fp16=False):
    if height % 32 or width % 32:
        raise ValueError("height/width должны быть кратны 32")
    os.makedirs(out_dir, exist_ok=True)

    cfg, model, num_classes = _load_model(cfg_path, ckpt_path)

    with torch.no_grad():
        probe = model.backbone(torch.randn(1, 3, 256, 256, device="cuda"))
    fpn_keys = list(probe.keys())
    print(f"[info] fpn_keys={fpn_keys}")

    dummy = torch.randn(1, 3, height, width, device="cuda")

    # ── граф 1 ──────────────────────────────────────────────────────
    g1 = Graph1_BackboneRPNHead(model, fpn_keys).eval()
    feat_names  = [f"feat_{k}"  for k in fpn_keys]
    obj_names   = [f"obj_{k}"   for k in fpn_keys]
    delta_names = [f"delta_{k}" for k in fpn_keys]
    out_names_g1 = feat_names + obj_names + delta_names

    dyn_g1 = None
    if not static:
        dyn_g1 = {"images": {0: "batch", 2: "h", 3: "w"}}
        # у каждого уровня свои символьные H/W
        for i, k in enumerate(fpn_keys):
            for n in (f"feat_{k}", f"obj_{k}", f"delta_{k}"):
                dyn_g1[n] = {0: "batch", 2: f"h{i}", 3: f"w{i}"}

    path_g1 = os.path.join(out_dir, "graph1_backbone_rpn.onnx")
    torch.onnx.export(
        g1, dummy, path_g1,
        input_names=["images"],
        output_names=out_names_g1,
        dynamic_axes=dyn_g1,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[ok] {path_g1}  ({os.path.getsize(path_g1) / 2**20:.1f} MB)")

    # ── граф 2 ──────────────────────────────────────────────────────
    pool = model.roi_heads.box_roi_pool
    pool_size = int(pool.output_size[0])
    out_ch = int(model.backbone.out_channels)
    dummy_pooled = torch.randn(10, out_ch, pool_size, pool_size, device="cuda")

    g2 = Graph2_BoxHead(model).eval()
    path_g2 = os.path.join(out_dir, "graph2_box_head.onnx")
    torch.onnx.export(
        g2, dummy_pooled, path_g2,
        input_names=["pooled"],
        output_names=["cls_logits", "box_regression"],
        dynamic_axes={
            "pooled":         {0: "n_prop"},
            "cls_logits":     {0: "n_prop"},
            "box_regression": {0: "n_prop"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[ok] {path_g2}  ({os.path.getsize(path_g2) / 2**20:.1f} MB)")

    if fp16:
        _to_fp16(path_g1)
        _to_fp16(path_g2)

    # ── мета ────────────────────────────────────────────────────────
    t, rpn, rh = model.transform, model.rpn, model.roi_heads
    meta = {
        "arch": cfg.model.arch,
        "num_classes": num_classes,
        "fpn_keys": fpn_keys,
        "out_channels": out_ch,
        "static": static,
        "export_shape": [height, width],
        "opset": opset,
        # transform
        "image_mean": [float(v) for v in t.image_mean],
        "image_std": [float(v) for v in t.image_std],
        "min_size": [int(v) for v in t.min_size],
        "max_size": int(t.max_size),
        "size_divisible": int(t.size_divisible),
        # RPN
        "anchor_sizes": [list(s) for s in rpn.anchor_generator.sizes],
        "aspect_ratios": [list(r) for r in rpn.anchor_generator.aspect_ratios],
        "rpn_pre_nms_top_n": int(rpn._pre_nms_top_n["testing"]),
        "rpn_post_nms_top_n": int(rpn._post_nms_top_n["testing"]),
        "rpn_nms_thresh": float(rpn.nms_thresh),
        "rpn_score_thresh": float(rpn.score_thresh),
        "rpn_min_size": float(rpn.min_size),
        # RoI
        "roi_featmap_names": list(pool.featmap_names),   # без 'pool'
        "roi_sampling_ratio": int(pool.sampling_ratio),
        "pool_size": pool_size,
        "box_coder_weights": [float(w) for w in rh.box_coder.weights],
        "score_thresh": float(rh.score_thresh),
        "nms_thresh": float(rh.nms_thresh),
        "detections_per_img": int(rh.detections_per_img),
        "torchvision": __import__("torchvision").__version__,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[ok] meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="директория для двух графов")
    p.add_argument("--height", type=int, default=160,
                   help="размер входа ПОСЛЕ transform (для --static должен совпадать с реальным)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--static", action="store_true", help="фиксированный размер входа")
    p.add_argument("--fp16", action="store_true", help="дополнительно сохранить fp16-графы")
    a = p.parse_args()
    export(a.config, a.ckpt, a.out, a.height, a.width, a.opset, a.static, a.fp16)
