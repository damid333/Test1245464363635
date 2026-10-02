"""Замеры скорости / памяти / точности на одном val-сплите и одним COCOeval.

Бэкенды:
  onnx   — наш Faster R-CNN, два графа             --export-dir
  onnx1  — наш Faster R-CNN, один граф             --export-dir
  torch  — наш Faster R-CNN в PyTorch (эталон)     --config --ckpt [--amp]
  yolo   — YOLO12 через ultralytics (.pt/.onnx)    --weights --imgsz [--half]
  mmdet  — текущий прод на mmdet (ResNet-50)       --config <mmdet .py> --ckpt <.pth>

Общие аргументы: --img-dir --ann-file --out [--device-id --warmup --speed-iters --skip-map]
Память GPU пишется при установленном nvidia-ml-py:  uv pip install nvidia-ml-py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import cv2
import numpy as np


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def class_names_to_cat_ids(names, ann_file):
    """Индекс класса модели -> category_id датасета. Сначала по именам, иначе по порядку id."""
    with open(ann_file) as f:
        cats = json.load(f)["categories"]
    by_name = {c["name"]: c["id"] for c in cats}
    names = list(names) if names is not None else []
    if names and all(n in by_name for n in names):
        mapping = {i: by_name[n] for i, n in enumerate(names)}
        print(f"[classes] по именам: {mapping}")
        return mapping
    ids = sorted(c["id"] for c in cats)
    mapping = {i: cid for i, cid in enumerate(ids)}
    print(f"[classes][warn] имена модели {names} не совпали с категориями {list(by_name)} — "
          f"сопоставляю по порядку: {mapping}. ПРОВЕРЬТЕ, что порядок верный!")
    return mapping


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


# ------------------------------------------------------------ наш Faster R-CNN в PyTorch
class TorchRunner:
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


# ------------------------------------------------------------ YOLO12 (ultralytics)
class YoloRunner:
    """ultralytics.YOLO: .pt, .onnx, .engine. Пре/постпроцессинг ultralytics входит в замер."""

    def __init__(self, weights, ann_file, device_id=0, imgsz=640, half=False, conf=0.05, iou=0.7):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=100, device=device_id,
                       half=half, verbose=False)
        names = self.model.names
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        self.cat_ids = class_names_to_cat_ids(names, ann_file)
        self.info = {"weights": weights, "imgsz": imgsz, "half": half, "conf": conf, "iou": iou,
                     "classes": names}
        self.last_timings = {}

    @property
    def providers(self):
        return [f"ultralytics:{self.kw['device']}"]

    def __call__(self, img_rgb):
        t0 = time.perf_counter()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # ultralytics ждёт BGR, как из cv2.imread
        r = self.model.predict(img_bgr, **self.kw)[0]
        b = r.boxes
        boxes = b.xyxy.cpu().numpy().astype(np.float32)        # .cpu() = синхронизация GPU
        scores = b.conf.cpu().numpy().astype(np.float32)
        labels = b.cls.cpu().numpy().astype(np.int64)
        res = {"boxes": boxes, "scores": scores, "labels": labels,
               "category_ids": np.array([self.cat_ids.get(int(l), -1) for l in labels], np.int64)}
        sp = r.speed or {}
        self.last_timings = {"total": time.perf_counter() - t0,
                             "pre": sp.get("preprocess", 0.0) / 1000,
                             "model": sp.get("inference", 0.0) / 1000,
                             "post": sp.get("postprocess", 0.0) / 1000}
        return res


# ------------------------------------------------------------ mmdet (текущий прод)
class MmdetRunner:
    """mmdet 3.x (DetDataSample) и 2.x (список массивов по классам)."""

    def __init__(self, config, ckpt, ann_file, device_id=0):
        from mmdet.apis import inference_detector, init_detector

        self.infer = inference_detector
        self.model = init_detector(config, ckpt, device=f"cuda:{device_id}")
        meta = getattr(self.model, "dataset_meta", None) or {}
        names = meta.get("classes") or getattr(self.model, "CLASSES", None)
        self.cat_ids = class_names_to_cat_ids(names, ann_file)
        self.info = {"config": config, "ckpt": ckpt, "classes": list(names) if names else None}
        self.last_timings = {}

    @property
    def providers(self):
        return ["mmdet:cuda"]

    def __call__(self, img_rgb):
        t0 = time.perf_counter()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # mmdet/mmcv считают ndarray как BGR
        r = self.infer(self.model, img_bgr)
        if hasattr(r, "pred_instances"):                     # mmdet 3.x
            pi = r.pred_instances
            boxes = pi.bboxes.cpu().numpy().astype(np.float32)
            scores = pi.scores.cpu().numpy().astype(np.float32)
            labels = pi.labels.cpu().numpy().astype(np.int64)
        else:                                                # mmdet 2.x
            per_class = r[0] if isinstance(r, tuple) else r
            boxes, scores, labels = [], [], []
            for cls, arr in enumerate(per_class):
                if len(arr):
                    boxes.append(arr[:, :4])
                    scores.append(arr[:, 4])
                    labels.append(np.full(len(arr), cls))
            if boxes:
                boxes = np.concatenate(boxes).astype(np.float32)
                scores = np.concatenate(scores).astype(np.float32)
                labels = np.concatenate(labels).astype(np.int64)
            else:
                boxes, scores, labels = np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        res = {"boxes": boxes, "scores": scores, "labels": labels,
               "category_ids": np.array([self.cat_ids.get(int(l), -1) for l in labels], np.int64)}
        self.last_timings = {"total": time.perf_counter() - t0}
        return res


# ------------------------------------------------------------ helpers
def make_runner(a):
    if a.backend == "onnx":
        from fcos_train.inference_onnx_final import OnnxDetector
        r = OnnxDetector(a.export_dir, device_id=a.device_id)
        info = {"export_dir": a.export_dir, "mode": "two_graphs", "providers": r.providers,
                "fp16": r.meta.get("fp16"), "canvas_hw": r.meta.get("canvas_hw")}
    elif a.backend == "onnx1":
        from fcos_train.inference_onnx_final import OnnxSingleDetector
        r = OnnxSingleDetector(a.export_dir, device_id=a.device_id)
        info = {"export_dir": a.export_dir, "mode": "single_graph", "providers": r.providers,
                "input_hw": r.meta.get("input_hw")}
    elif a.backend == "torch":
        r = TorchRunner(a.config, a.ckpt, a.device_id, a.amp)
        info = {"ckpt": a.ckpt, "amp": a.amp, "device": r.dev}
    elif a.backend == "yolo":
        if not (a.weights and a.imgsz):
            raise SystemExit("--backend yolo требует --weights и --imgsz (тот, на котором обучали)")
        r = YoloRunner(a.weights, a.ann_file, a.device_id, a.imgsz, a.half, a.conf, a.iou)
        info = r.info
    else:  # mmdet
        if not (a.config and a.ckpt):
            raise SystemExit("--backend mmdet требует --config (mmdet .py) и --ckpt")
        r = MmdetRunner(a.config, a.ckpt, a.ann_file, a.device_id)
        info = r.info
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
            "AP_large": round(float(s[5]), 4), "AR100": round(float(s[8]), 4)}


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["onnx", "onnx1", "torch", "yolo", "mmdet"], required=True)
    ap.add_argument("--export-dir", help="onnx / onnx1")
    ap.add_argument("--config", help="torch: наш config.yaml; mmdet: конфиг mmdet .py")
    ap.add_argument("--ckpt", help="torch / mmdet: веса .pth")
    ap.add_argument("--amp", action="store_true", help="torch: autocast fp16")
    ap.add_argument("--weights", help="yolo: .pt / .onnx / .engine")
    ap.add_argument("--imgsz", type=int, help="yolo: размер входа, как при обучении")
    ap.add_argument("--half", action="store_true", help="yolo: fp16")
    ap.add_argument("--conf", type=float, default=0.05, help="yolo: порог score (0.05 как у Faster R-CNN)")
    ap.add_argument("--iou", type=float, default=0.7, help="yolo: порог NMS")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--speed-images", type=int, default=100)
    ap.add_argument("--speed-iters", type=int, default=300)
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args()
    if a.backend in ("onnx", "onnx1") and not a.export_dir:
        raise SystemExit("--backend onnx/onnx1 требует --export-dir")
    if a.backend == "torch" and not (a.config and a.ckpt):
        raise SystemExit("--backend torch требует --config и --ckpt")

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
        unknown = 0
        for img_id, path in val:
            r = runner(read_rgb(path))
            for b, s, c in zip(r["boxes"], r["scores"], r["category_ids"]):
                if c < 0:
                    unknown += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in b)
                dets.append({"image_id": img_id, "category_id": int(c),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if unknown:
            print(f"[warn] {unknown} детекций с неизвестным классом пропущены — проверьте маппинг классов")
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
