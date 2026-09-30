"""Прод-инференс двухграфовой модели. Зависимости: onnxruntime-gpu, numpy, opencv. Torch не нужен.

    det = OnnxDetector("export/exp_X", device_id=0)
    res = det(rgb_uint8_image)   # {"boxes": (D,4) xyxy в пикселях оригинала, "scores", "labels", "category_ids"}
"""
from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    def __init__(self, export_dir: str, device_id: int = 0, cudnn_search: str = "EXHAUSTIVE",
                 intra_op_threads: int = 0):
        with open(os.path.join(export_dir, "meta.json")) as f:
            self.meta = m = json.load(f)
        self.H, self.W = m["canvas_hw"]
        self.min_size, self.max_size = m["min_size"], m["max_size"]
        self.mean = np.asarray(m["image_mean"], np.float32)
        self.std = np.asarray(m["image_std"], np.float32)
        self.feat_names = m["feat_names"]
        self.cat_ids = {int(k): int(v) for k, v in m["contiguous_to_cat_id"].items()}

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        if "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_opts = {
                "device_id": device_id,
                "cudnn_conv_algo_search": cudnn_search,
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": True,
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
            self.dev, self.dev_id = "cuda", device_id
        else:
            print("[OnnxDetector] CUDAExecutionProvider недоступен — работаю на CPU")
            providers, self.dev, self.dev_id = ["CPUExecutionProvider"], "cpu", 0

        self.s1 = ort.InferenceSession(os.path.join(export_dir, "stage1.onnx"), so, providers=providers)
        self.s2 = ort.InferenceSession(os.path.join(export_dir, "stage2.onnx"), so, providers=providers)
        self.s1_out = [o.name for o in self.s1.get_outputs()]   # feat0..featN, proposals
        self.s2_out = [o.name for o in self.s2.get_outputs()]   # boxes, scores, labels
        self.canvas = np.zeros((1, 3, self.H, self.W), np.float32)
        self.last_timings: dict = {}

    @property
    def providers(self):
        return self.s1.get_providers()

    # ------------------------------------------------------------ pre / post
    def preprocess(self, img_rgb: np.ndarray):
        """Повторяет GeneralizedRCNNTransform (eval): resize по min/max стороне, normalize, паддинг вправо-вниз."""
        h, w = img_rgb.shape[:2]
        scale = float(min(np.float32(self.min_size) / np.float32(min(h, w)),
                          np.float32(self.max_size) / np.float32(max(h, w))))
        nh, nw = int(math.floor(h * scale)), int(math.floor(w * scale))
        if nh > self.H or nw > self.W:
            raise ValueError(f"после ресайза {nh}x{nw} не влезает в canvas {self.H}x{self.W}; "
                             f"переэкспортируйте с другим --canvas")
        img = cv2.resize(img_rgb.astype(np.float32) * (1.0 / 255.0), (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = (img - self.mean) / self.std
        self.canvas.fill(0.0)
        self.canvas[0, :, :nh, :nw] = img.transpose(2, 0, 1)
        return self.canvas, np.array([nh, nw], np.float32), (h / nh, w / nw)

    def _empty(self):
        return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
                "labels": np.zeros(0, np.int64), "category_ids": np.zeros(0, np.int64)}

    # ------------------------------------------------------------ run
    def __call__(self, img_rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        x, hw, (ry, rx) = self.preprocess(img_rgb)
        t1 = time.perf_counter()

        io1 = self.s1.io_binding()
        io1.bind_cpu_input("image", x)
        io1.bind_cpu_input("image_hw", hw)
        for n in self.s1_out:                         # признаки и proposals остаются на GPU
            io1.bind_output(n, self.dev, self.dev_id)
        self.s1.run_with_iobinding(io1)
        outs1 = dict(zip(self.s1_out, io1.get_outputs()))
        t2 = time.perf_counter()

        if outs1["proposals"].shape()[0] == 0:
            self.last_timings = {"pre": t1 - t0, "stage1": t2 - t1, "stage2": 0.0, "post": 0.0, "total": t2 - t0}
            return self._empty()

        io2 = self.s2.io_binding()
        for n in self.feat_names:
            io2.bind_ortvalue_input(n, outs1[n])
        io2.bind_ortvalue_input("rois", outs1["proposals"])
        io2.bind_cpu_input("image_hw", hw)
        for n in self.s2_out:
            io2.bind_output(n, "cpu")
        self.s2.run_with_iobinding(io2)
        res = dict(zip(self.s2_out, (v.numpy() for v in io2.get_outputs())))
        t3 = time.perf_counter()

        boxes = res["boxes"].astype(np.float32)
        boxes[:, 0::2] *= rx
        boxes[:, 1::2] *= ry
        labels = res["labels"].astype(np.int64)
        out = {"boxes": boxes, "scores": res["scores"].astype(np.float32), "labels": labels,
               "category_ids": np.array([self.cat_ids.get(int(l), -1) for l in labels], np.int64)}
        t4 = time.perf_counter()
        self.last_timings = {"pre": t1 - t0, "stage1": t2 - t1, "stage2": t3 - t2, "post": t4 - t3, "total": t4 - t0}
        return out


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--save", default=None, help="сохранить картинку с боксами")
    a = ap.parse_args()

    det = OnnxDetector(a.export_dir)
    print("providers:", det.providers)
    img = read_rgb(a.image)
    r = det(img)
    print({k: f"{v * 1000:.1f} ms" for k, v in det.last_timings.items()})
    for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
        if s >= a.thr:
            print(f"cat={c} score={s:.3f} box={np.round(b, 1).tolist()}")
    if a.save:
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
            if s >= a.thr:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{c}:{s:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(a.save, vis)
