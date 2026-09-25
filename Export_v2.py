# src/fcos_train/export_two_graphs.py
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList

from .config import load_config
from .model import build_model


class Graph1_BackboneRPN(nn.Module):
    """backbone + FPN + RPN -> фичемапы + proposals. Всё на GPU."""

    def __init__(self, model, fpn_keys):
        super().__init__()
        self.backbone = model.backbone
        self.rpn = model.rpn
        self.fpn_keys = fpn_keys

    def forward(self, images: torch.Tensor):
        h, w = int(images.shape[-2]), int(images.shape[-1])
        image_list = ImageList(images, [(h, w)])
        feats = self.backbone(images)
        proposals, _ = self.rpn(image_list, feats, None)
        # кортеж фиксированного порядка — трассировщик не любит dict
        feat_tuple = tuple(feats[k] for k in self.fpn_keys)
        return feat_tuple + (proposals[0],)   # batch=1


class Graph2_BoxHead(nn.Module):
    """box_head + box_predictor -> logits + regression. Всё на GPU."""

    def __init__(self, model):
        super().__init__()
        self.box_head = model.roi_heads.box_head
        self.box_predictor = model.roi_heads.box_predictor

    def forward(self, pooled: torch.Tensor):
        feat = self.box_head(pooled)
        cls_logits, box_reg = self.box_predictor(feat)
        return cls_logits, box_reg


def export(cfg_path, ckpt_path, out_dir,
           height=800, width=1344, opset=16):
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    model = build_model(cfg.model, num_classes)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval().cuda()

    # снимаем ключи FPN с живой модели
    with torch.no_grad():
        probe = model.backbone(torch.randn(1, 3, 256, 256, device="cuda"))
    fpn_keys = list(probe.keys())
    print(f"[info] fpn_keys={fpn_keys}")

    dummy = torch.randn(1, 3, height, width, device="cuda")

    # ── граф 1 ──────────────────────────────────────────────────────
    g1 = Graph1_BackboneRPN(model, fpn_keys).eval()
    feat_names = [f"feat_{k}" for k in fpn_keys]
    out_names_g1 = feat_names + ["proposals"]

    dyn_g1 = {"images": {0: "batch", 2: "h", 3: "w"}}
    for n in feat_names:
        dyn_g1[n] = {0: "batch", 2: "fh", 3: "fw"}
    dyn_g1["proposals"] = {0: "n_prop"}

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
    print(f"[ok] {path_g1}  ({os.path.getsize(path_g1)/2**20:.1f} MB)")

    # ── граф 2 ──────────────────────────────────────────────────────
    # dummy pooled: [N, out_channels, 7, 7]
    pool_size = model.roi_heads.box_roi_pool.output_size[0]
    out_ch = model.backbone.out_channels
    dummy_pooled = torch.randn(10, out_ch, pool_size, pool_size, device="cuda")

    g2 = Graph2_BoxHead(model).eval()
    path_g2 = os.path.join(out_dir, "graph2_box_head.onnx")
    torch.onnx.export(
        g2, dummy_pooled, path_g2,
        input_names=["pooled"],
        output_names=["cls_logits", "box_regression"],
        dynamic_axes={
            "pooled":        {0: "n_prop"},
            "cls_logits":    {0: "n_prop"},
            "box_regression": {0: "n_prop"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[ok] {path_g2}  ({os.path.getsize(path_g2)/2**20:.1f} MB)")

    # ── мета ────────────────────────────────────────────────────────
    t = model.transform
    meta = {
        "arch": cfg.model.arch,
        "num_classes": num_classes,
        "fpn_keys": fpn_keys,
        "out_channels": int(out_ch),
        "pool_size": int(pool_size),
        "image_mean": list(t.image_mean),
        "image_std": list(t.image_std),
        "min_size": list(t.min_size),
        "max_size": int(t.max_size),
        "size_divisible": 32,
        "score_thresh": float(model.roi_heads.score_thresh),
        "nms_thresh": float(model.roi_heads.nms_thresh),
        "detections_per_img": int(model.roi_heads.detections_per_img),
        "torchvision": __import__("torchvision").__version__,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--out",    required=True, help="директория для двух графов")
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--width",  type=int, default=1344)
    p.add_argument("--opset",  type=int, default=16)
    a = p.parse_args()
    export(a.config, a.ckpt, a.out, a.height, a.width, a.opset)
