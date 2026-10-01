"""Замеры скорости / памяти / точности: ONNX (два графа или один) vs PyTorch.

Два графа:
    python -m fcos_train.bench_final --backend onnx --export-dir .../onnx \
        --img-dir .../val --ann-file .../instances_val.json --out report_onnx.json
Один граф (классический экспорт):
    python -m fcos_train.bench_final --backend onnx1 --export-dir .../onnx_single \
        --img-dir .../val --ann-file .../instances_val.json --out report_onnx_single.json
PyTorch (эталон):
    python -m fcos_train.bench_final --backend torch --config .../config.yaml --ckpt .../best.pth \
        --img-dir .../val --ann-file .../instances_val.json --out report_torch.json [--amp]

Память GPU пишется, если установлен nvidia-ml-py:  uv pip install nvidia-ml-py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from fcos_train.inference_onnx_final import OnnxDetector, OnnxSingleDetector, read_rgb


# ------------------------------------------------------------ GPU memory (nvml)
class GpuMem:
    def __init__(self, device_id=0):
        self.ok = False
        self.name = None
        self.peak_proc = 0
        self.peak_total = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(self.h)
            self.name = name.decode() if isinstance(name, bytes) else name
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[mem] pynvml недоступен ({e}) — память не замеряется")

    def sample(self):
        if not self.ok:
            return
        pid = os.getpid()
        for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.h):
            if p.pid == pid and p.usedGpuMemory:
                self.peak_proc = max(self.peak_proc, p.usedGpuMemory)
        self.peak_total = max(self.peak_total, self.nvml.nvmlDeviceGetMemoryInfo(self.h).used)

    def report(self):
        mb = lambda b: round(b / 2**20, 1) if b else None  # noqa: E731
        return {"gpu": self.name, "process_peak_mb": mb(self.peak_proc),
                "device_used_peak_mb": mb(self.peak_total)}


# ------------------------------------------------------------ PyTorch эталон
class TorchRunner:
    """Исходная модель целиком, со своим transform и постпроцессингом."""

    def __init__(self, config, ckpt, device_id=0, amp=False):
        import torch
        from fcos_train.export_final import load_model

        self.torch = torch
        self.dev = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.model, _, _, mapping = load_model(config, ckpt, self.dev)
        self.cat_ids = {v: k for k, v in mapping.items()}
        self.amp = amp and self.dev.startswith("cuda")
        self.last_timings = {}
        torch.backends.cudnn.benchmark = True

    @property
    def providers(self):
        return [self.dev]

    def __call__(self, img_rgb):
        torch = self.torch
        t0 = time.perf_counter()
        x = torch.from_numpy(img_rgb).to(self.dev).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.amp):
            out = self.model([x])[0]
        if self.dev.startswith("cuda"):
            torch.cuda.synchronize()
        labels = out["labels"].cpu().numpy().astype(np.int64)
        res = {"boxes": out["boxes"].float().cpu().numpy(),
               "scores": out["scores"].float().cpu().numpy(),
               "labels": labels,
               "category_ids": np.array([self.cat_ids.get(int(l), -1) for l in labels], np.int64)}
        self.last_timings = {"total": time.perf_counter() - t0}
        return res


# ------------------------------------------------------------ helpers
def make_runner(a):
    if a.backend == "onnx":
        r = OnnxDetector(a.export_dir, device_id=a.device_id)
        info = {"export_dir": a.export_dir, "mode": "two_graphs", "providers": r.providers,
                "fp16": r.meta.get("fp16"), "canvas_hw": r.meta.get("canvas_hw")}
    elif a.backend == "onnx1":
        r = OnnxSingleDetector(a.export_dir, device_id=a.device_id)
        info = {"export_dir": a.export_dir, "mode": "single_graph", "providers": r.providers,
                "input_hw": r.meta.get("input_hw")}
    else:
        if not (a.config and a.ckpt):
            raise SystemExit("--backend torch требует --config и --ckpt")
        r = TorchRunner(a.config, a.ckpt, a.device_id, a.amp)
        info = {"ckpt": a.ckpt, "amp": a.amp, "device": r.dev}
    return r, info


def load_val(ann_file, img_dir, limit=None):
    with open(ann_file) as f:
        coco = json.load(f)
    imgs = coco["images"][:limit] if limit else coco["images"]
    return [(im["id"], os.path.join(img_dir, im["file_name"])) for im in imgs]


def stats(xs):
    a = np.asarray(xs) * 1000.0
    return {"mean_ms": round(float(a.mean()), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "min_ms": round(float(a.min()), 2),
            "max_ms": round(float(a.max()), 2)}


def coco_map(ann_file, detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}
    gt = COCO(ann_file)
    dt = gt.loadRes(detections)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {"mAP": round(float(s[0]), 4), "AP50": round(float(s[1]), 4), "AP75": round(float(s[2]), 4),
            "AP_small": round(float(s[3]), 4), "AP_medium": round(float(s[4]), 4),
            "AP_large": round(float(s[5]), 4)}


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["onnx", "onnx1", "torch"], required=True,
                    help="onnx — два графа, onnx1 — один граф, torch — эталон")
    ap.add_argument("--export-dir")
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--amp", action="store_true", help="torch: autocast fp16")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--speed-images", type=int, default=100, help="картинок в памяти для замера скорости")
    ap.add_argument("--speed-iters", type=int, default=300)
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args()
    if a.backend in ("onnx", "onnx1") and not a.export_dir:
        raise SystemExit("--backend onnx/onnx1 требует --export-dir")

    mem = GpuMem(a.device_id)
    t_init = time.perf_counter()
    runner, backend_info = make_runner(a)
    init_s = time.perf_counter() - t_init
    print("providers:", runner.providers)
    mem.sample()

    val = load_val(a.ann_file, a.img_dir)

    # ---- скорость: картинки заранее в памяти, диск не мерим
    imgs = [read_rgb(p) for _, p in val[: a.speed_images]]
    for i in range(a.warmup):
        runner(imgs[i % len(imgs)])
    mem.sample()

    per_stage: dict[str, list] = {}
    t_start = time.perf_counter()
    for i in range(a.speed_iters):
        runner(imgs[i % len(imgs)])
        for k, v in runner.last_timings.items():
            per_stage.setdefault(k, []).append(v)
        if i % 20 == 0:
            mem.sample()
    wall = time.perf_counter() - t_start

    speed = {k: stats(v) for k, v in per_stage.items()}
    tot = np.asarray(per_stage["total"]) * 1000.0
    speed["slow_frames_gt_2x_median"] = int((tot > 2 * np.median(tot)).sum())
    speed["throughput_img_s"] = round(a.speed_iters / wall, 2)
    speed["init_s"] = round(init_s, 2)

    # ---- точность на всём val
    acc = None
    if not a.skip_map:
        dets = []
        for img_id, path in val:
            r = runner(read_rgb(path))
            for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
                x1, y1, x2, y2 = (float(v) for v in b)
                dets.append({"image_id": img_id, "category_id": int(c),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        mem.sample()
        acc = coco_map(a.ann_file, dets)
        acc["num_images"] = len(val)
        acc["num_detections"] = len(dets)

    report = {"backend": a.backend, **backend_info, "host": platform.node(),
              "speed": speed, "memory": mem.report(), "accuracy": acc}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
