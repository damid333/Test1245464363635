# src/fcos_train/export_frcnn_onnx.py
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList

from .config import load_config
from .model import build_model


class FrcnnGraph(nn.Module):
    """backbone + FPN + RPN + roi_heads. Batch = 1.

    Вход: нормализованный NCHW, кратный 32. Выход: боксы в координатах
    паддингованного тензора, обратный скейл к оригиналу делается снаружи.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads

    def forward(self, images: torch.Tensor):
        h, w = int(images.shape[-2]), int(images.shape[-1])
        image_list = ImageList(images, [(h, w)])
        feats = self.backbone(images)
        proposals, _ = self.rpn(image_list, feats, None)
        dets, _ = self.roi_heads(feats, proposals, image_list.image_sizes, None)
        d = dets[0]
        return d["boxes"], d["scores"], d["labels"]


def export(cfg_path: str, ckpt_path: str, out_path: str,
           height: int = 800, width: int = 1344, opset: int = 16) -> None:
    cfg = load_config(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    if cfg.model.arch not in ("faster_rcnn", "cascade_rcnn"):
        raise ValueError(f"ожидался faster_rcnn/cascade_rcnn, получено {cfg.model.arch}")
    if height % 32 or width % 32:
        raise ValueError("размер должен быть кратен 32")
    if opset < 16:
        # RoiAlign с aligned=True (рецепт v2) требует ONNX RoiAlign-16
        raise ValueError("нужен opset >= 16 из-за aligned RoiAlign")

    model = build_model(cfg.model, num_classes)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    dummy = torch.randn(1, 3, height, width)
    torch.onnx.export(
        FrcnnGraph(model).eval(), dummy, out_path,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        # число детекций меняется от картинки к картинке
        dynamic_axes={"boxes": {0: "n"}, "scores": {0: "n"}, "labels": {0: "n"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    t = model.transform
    meta = {
        "arch": cfg.model.arch,
        "num_classes": num_classes,
        "export_shape": [height, width],
        "opset": opset,
        "image_mean": list(t.image_mean),
        "image_std": list(t.image_std),
        "min_size": list(t.min_size),
        "max_size": int(t.max_size),
        "size_divisible": 32,
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
    p.add_argument("--opset", type=int, default=16)
    a = p.parse_args()
    export(a.config, a.ckpt, a.out, a.height, a.width, a.opset)
