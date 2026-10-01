"""Классический экспорт: вся модель (transform + RPN + RoI + postprocess) одним графом.

Вход:  image — RGB float32 (3, H, W) в [0, 1], ровно того размера, на котором трассировали.
Выход: boxes (N, 4) в пикселях оригинала, scores (N), labels (N).

python -m fcos_train.export_single --config .../config.yaml --ckpt .../best.pth \
    --image .../00114550_05_00458.jpg --out .../onnx_single
"""
from __future__ import annotations

import argparse
import inspect
import json
import os

import cv2
import numpy as np
import torch

from fcos_train.export_final import load_model


class Wrapper(torch.nn.Module):
    """Модель целиком; на выходе кортеж вместо list[dict]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):                     # (3, H, W)
        out = self.model([image])[0]
        return out["boxes"], out["scores"], out["labels"]


def read_tensor(path, device):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True, help="реальный кадр: по нему трассировка и размер входа")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model, _, num_classes, mapping = load_model(a.config, a.ckpt, a.device)
    w = Wrapper(model).eval()
    x = read_tensor(a.image, a.device)
    H, W = x.shape[-2:]

    with torch.no_grad():
        ref = [t.cpu().numpy() for t in w(x)]

    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    path = os.path.join(a.out, "model.onnx")
    torch.onnx.export(
        w, (x,), path, opset_version=a.opset,
        input_names=["image"], output_names=["boxes", "scores", "labels"],
        dynamic_axes={"boxes": {0: "N"}, "scores": {0: "N"}, "labels": {0: "N"}},
        do_constant_folding=True, **kw,
    )
    meta = {"mode": "single", "input_hw": [int(H), int(W)], "num_classes": num_classes,
            "contiguous_to_cat_id": {str(v): k for k, v in mapping.items()}, "opset": a.opset}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[export] {path}  вход 3x{H}x{W}")

    # --- сверка с torch на том же кадре + проверка пустого кадра ---
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("[verify] providers:", sess.get_providers())
    got = sess.run(None, {"image": x.cpu().numpy()})
    print(f"[verify] детекций torch={len(ref[1])} onnx={len(got[1])}")
    n = min(len(ref[1]), len(got[1]), 10)
    if n:
        print(f"[verify] top-{n} max|Δbox|={np.abs(ref[0][:n] - got[0][:n]).max():.3e} "
              f"max|Δscore|={np.abs(ref[1][:n] - got[1][:n]).max():.3e}")
    try:
        empty = sess.run(None, {"image": np.zeros((3, H, W), np.float32)})
        print(f"[verify] пустой кадр: OK, детекций {len(empty[1])}")
    except Exception as e:  # noqa: BLE001
        print(f"[verify] пустой кадр: ПАДАЕТ — {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
