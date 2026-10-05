"""Прод-инференс ONNX-модели. Зависимости: onnxruntime-gpu[cuda,cudnn], numpy, opencv.
Для ep=trt / trt_fp16 дополнительно: tensorrt-cu13.

Два графа (stage1.onnx + stage2.onnx):  OnnxDetector(dir, ep="cuda")
Один граф (model.onnx):                 OnnxSingleDetector(dir, ep="cuda" | "trt" | "trt_fp16")
Автовыбор по meta.json:                 load_detector(dir, ep=...)

    res = det(rgb_uint8_image)
    # {"boxes": (D,4) xyxy в пикселях оригинала, "scores", "labels", "category_ids"}
"""
from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# CUDA/cuDNN из пакетов nvidia-* (onnxruntime-gpu[cuda,cudnn]); без этого сессия падает на CPU
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

EPS = ("cuda", "trt", "trt_fp16")


def _session_options(intra_op_threads: int = 0, verbose: bool = False) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads:
        so.intra_op_num_threads = intra_op_threads
    if verbose:
        so.log_severity_level = 1   # покажет, какие узлы ушли в TensorRT, а какие в CUDA/CPU
    return so


def _providers(device_id: int, cudnn_search: str, ep: str = "cuda", cache_dir: str | None = None):
    """Возвращает (providers, куда биндить выходы, device_id)."""
    if ep not in EPS:
        raise ValueError(f"ep={ep!r}, допустимо: {EPS}")
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        print("[onnx] CUDAExecutionProvider недоступен — работаю на CPU")
        return ["CPUExecutionProvider"], "cpu", 0

    cuda_opts = {
        "device_id": device_id,
        "cudnn_conv_algo_search": cudnn_search,
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": True,
    }
    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

    if ep.startswith("trt"):
        try:
            import tensorrt  # noqa: F401  — подгружает libnvinfer в процесс
        except ImportError:
            print("[onnx] пакет tensorrt не найден — TensorRT EP может не подняться")
        if "TensorrtExecutionProvider" not in avail:
            raise RuntimeError("TensorrtExecutionProvider нет в onnxruntime: проверьте onnxruntime-gpu и tensorrt")
        cache_dir = cache_dir or "trt_cache"
        os.makedirs(cache_dir, exist_ok=True)
        trt_opts = {
            "device_id": device_id,
            "trt_fp16_enable": ep == "trt_fp16",
            "trt_engine_cache_enable": True,        # движок строится один раз и кладётся на диск
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_dir,
            "trt_max_workspace_size": 4 * 1024**3,
        }
        providers = [("TensorrtExecutionProvider", trt_opts)] + providers
    return providers, "cuda", device_id


def _empty() -> dict:
    return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
            "labels": np.zeros(0, np.int64), "category_ids": np.zeros(0, np.int64)}


# =====================================================================================
# Два графа: stage1 (backbone+FPN+RPN+NMS) -> stage2 (RoIAlign+head+NMS)
# =====================================================================================
class OnnxDetector:
    def __init__(self, export_dir: str, device_id: int = 0, cudnn_search: str = "EXHAUSTIVE",
                 intra_op_threads: int = 0, ep: str = "cuda", verbose: bool = False):
        with open(os.path.join(export_dir, "meta.json")) as f:
            self.meta = m = json.load(f)
        self.H, self.W = m["canvas_hw"]
        self.min_size, self.max_size = m["min_size"], m["max_size"]
        self.mean = np.asarray(m["image_mean"], np.float32)
        self.std = np.asarray(m["image_std"], np.float32)
        self.feat_names = m["feat_names"]
        self.cat_ids = {int(k): int(v) for k, v in m["contiguous_to_cat_id"].items()}
        self.ep = ep

        so = _session_options(intra_op_threads, verbose)
        providers, self.dev, self.dev_id = _providers(
            device_id, cudnn_search, ep, os.path.join(export_dir, f"trt_cache_{ep}"))
        self.s1 = ort.InferenceSession(os.path.join(export_dir, "stage1.onnx"), so, providers=providers)
        self.s2 = ort.InferenceSession(os.path.join(export_dir, "stage2.onnx"), so, providers=providers)
        self.s1_out = [o.name for o in self.s1.get_outputs()]   # feat0..featN, proposals
        self.s2_out = [o.name for o in self.s2.get_outputs()]   # boxes, scores, labels
        self.canvas = np.zeros((1, 3, self.H, self.W), np.float32)
        self.last_timings: dict = {}

    @property
    def providers(self):
        return self.s1.get_providers()

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

    def __call__(self, img_rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        x, hw, (ry, rx) = self.preprocess(img_rgb)
        t1 = time.perf_counter()

        io1 = self.s1.io_binding()
        io1.bind_cpu_input("image", x)
        io1.bind_cpu_input("image_hw", hw)
        for n in self.s1_out:
            io1.bind_output(n, self.dev, self.dev_id)
        self.s1.run_with_iobinding(io1)
        outs1 = dict(zip(self.s1_out, io1.get_outputs()))
        t2 = time.perf_counter()

        if outs1["proposals"].shape()[0] == 0:
            self.last_timings = {"pre": t1 - t0, "stage1": t2 - t1, "stage2": 0.0, "post": 0.0, "total": t2 - t0}
            return _empty()

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


# =====================================================================================
# Один граф: transform + RPN + RoI + postprocess внутри model.onnx
# =====================================================================================
class OnnxSingleDetector:
    def __init__(self, export_dir: str, device_id: int = 0, cudnn_search: str = "EXHAUSTIVE",
                 intra_op_threads: int = 0, ep: str = "cuda", verbose: bool = False):
        with open(os.path.join(export_dir, "meta.json")) as f:
            self.meta = m = json.load(f)
        self.H, self.W = m["input_hw"]
        self.cat_ids = {int(k): int(v) for k, v in m["contiguous_to_cat_id"].items()}
        self.ep = ep

        so = _session_options(intra_op_threads, verbose)
        providers, self.dev, self.dev_id = _providers(
            device_id, cudnn_search, ep, os.path.join(export_dir, f"trt_cache_{ep}"))
        self.sess = ort.InferenceSession(os.path.join(export_dir, "model.onnx"), so, providers=providers)
        self.out_names = [o.name for o in self.sess.get_outputs()]   # boxes, scores, labels
        self.last_timings: dict = {}

    @property
    def providers(self):
        return self.sess.get_providers()

    def __call__(self, img_rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        if img_rgb.shape[:2] != (self.H, self.W):
            raise ValueError(f"кадр {img_rgb.shape[:2]}, а граф собран под {(self.H, self.W)}")
        x = np.ascontiguousarray(img_rgb.transpose(2, 0, 1), dtype=np.float32) * (1.0 / 255.0)
        t1 = time.perf_counter()
        res = dict(zip(self.out_names, self.sess.run(None, {"image": x})))
        t2 = time.perf_counter()
        labels = res["labels"].astype(np.int64)
        out = {"boxes": res["boxes"].astype(np.float32), "scores": res["scores"].astype(np.float32),
               "labels": labels,
               "category_ids": np.array([self.cat_ids.get(int(l), -1) for l in labels], np.int64)}
        self.last_timings = {"pre": t1 - t0, "model": t2 - t1, "total": time.perf_counter() - t0}
        return out


# =====================================================================================
def load_detector(export_dir: str, device_id: int = 0, ep: str = "cuda", verbose: bool = False):
    """Выбирает класс по meta.json: mode=single -> один граф, иначе два."""
    with open(os.path.join(export_dir, "meta.json")) as f:
        mode = json.load(f).get("mode", "two_graphs")
    cls = OnnxSingleDetector if mode == "single" else OnnxDetector
    return cls(export_dir, device_id=device_id, ep=ep, verbose=verbose)


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
    ap.add_argument("--ep", choices=EPS, default="cuda")
    ap.add_argument("--verbose", action="store_true", help="лог onnxruntime: какие узлы на каком провайдере")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    t = time.perf_counter()
    det = load_detector(a.export_dir, ep=a.ep, verbose=a.verbose)
    print(f"класс: {type(det).__name__} | ep: {a.ep} | providers: {det.providers} | "
          f"создание сессии: {time.perf_counter() - t:.1f} с")
    img = read_rgb(a.image)

    r = _empty()
    for i in range(max(1, a.repeat)):
        r = det(img)
        if i in (0, 1, a.repeat - 1):
            print(f"прогон {i}:", {k: f"{v * 1000:.1f} ms" for k, v in det.last_timings.items()})

    for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
        if s >= a.thr:
            print(f"cat={c} score={s:.3f} box={np.round(b, 1).tolist()}")

    if a.save:
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
            if s >= a.thr:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{c}:{s:.2f}", (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(a.save, vis)
        print("сохранено:", a.save)
