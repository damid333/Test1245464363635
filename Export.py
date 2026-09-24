# src/fcos_train/export_fcos_onnx.py
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

from .config import load_config
from .model import build_model


class FcosGraph(nn.Module):
    """backbone + FPN + FCOS head -> три плоских тензора.

    FCOSHead уже конкатенирует уровни внутри себя, поэтому на выходе
    (N, sum(H_l*W_l), C) и никаких списков, которые ломают трассировку.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor):
        feats = list(self.backbone(images).values())
        out = self.head(feats)
        return out["cls_logits"], out["bbox_regression"], out["bbox_ctrness"]


def export(cfg_path: str, ckpt_path: str, out_path: str,
           height: int = 800, width: int = 1344,
           opset: int = 17, dynamic: bool = False) -> None:
    cfg = load_config(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    if cfg.model.arch != "fcos":
        raise ValueError(f"этот экспортёр только для arch: fcos, получено {cfg.model.arch}")

    model = build_model(cfg.model, num_classes)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    if height % 32 or width % 32:
        raise ValueError("размер должен быть кратен 32 (size_divisible трансформа)")

    dummy = torch.randn(1, 3, height, width)
    with torch.no_grad():
        feats = model.backbone(dummy)
        level_shapes = [[int(f.shape[-2]), int(f.shape[-1])] for f in feats.values()]
        cls, reg, ctr = FcosGraph(model)(dummy)

    napl = [h * w for h, w in level_shapes]
    assert sum(napl) == cls.shape[1], (napl, cls.shape)
    print(f"[info] levels={level_shapes} anchors_per_level={napl} "
          f"total={cls.shape[1]} num_classes={cls.shape[2]}")

    dyn = None
    if dynamic:
        dyn = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "cls_logits": {0: "batch", 1: "anchors"},
            "bbox_regression": {0: "batch", 1: "anchors"},
            "bbox_ctrness": {0: "batch", 1: "anchors"},
        }

    torch.onnx.export(
        FcosGraph(model).eval(), dummy, out_path,
        input_names=["images"],
        output_names=["cls_logits", "bbox_regression", "bbox_ctrness"],
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
    )

    ag = model.anchor_generator
    meta = {
        "arch": "fcos",
        "backbone": cfg.model.backbone,
        "num_classes": num_classes,
        "export_shape": [height, width],
        "dynamic": dynamic,
        "opset": opset,
        "level_shapes": level_shapes,
        "anchors_per_level": napl,
        "strides": [height // h for h, _ in level_shapes],
        "anchor_sizes": [list(s) for s in ag.sizes],
        "aspect_ratios": [list(r) for r in ag.aspect_ratios],
        "image_mean": list(model.transform.image_mean),
        "image_std": list(model.transform.image_std),
        "min_size": list(model.transform.min_size),
        "max_size": int(model.transform.max_size),
        "size_divisible": 32,
        "score_thresh": float(model.score_thresh),
        "nms_thresh": float(model.nms_thresh),
        "detections_per_img": int(model.detections_per_img),
        "topk_candidates": int(model.topk_candidates),
        "torchvision": __import__("torchvision").__version__,
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] {out_path} ({os.path.getsize(out_path) / 2**20:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic", action="store_true")
    a = p.parse_args()
    export(a.config, a.ckpt, a.out, a.height, a.width, a.opset, a.dynamic)
