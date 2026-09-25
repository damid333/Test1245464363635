# src/fcos_train/infer_onnx.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.dlpack import from_dlpack
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class FrcnnOnnxInfer:
    """
    Загружает frcnn.onnx, прогоняет изображения, возвращает детекции
    в координатах оригинального изображения.

    Тензоры не гоняются через CPU: вход биндится по data_ptr,
    выходы забираются через dlpack.
    """

    def __init__(self, onnx_path: str, device_id: int = 0):
        meta_path = onnx_path.replace(".onnx", ".json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            onnx_path, so,
            providers=[
                ("CUDAExecutionProvider", {
                    "device_id": device_id,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",   # статический shape — ок
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ],
        )

        got = self.sess.get_providers()
        print(f"[onnx] providers: {got}")
        if "CUDAExecutionProvider" not in got:
            raise RuntimeError(
                "CUDA EP не поднялся — сессия ушла на CPU. "
                "Проверь onnxruntime-gpu и версию CUDA."
            )

        self.in_name = self.sess.get_inputs()[0].name
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")

        m = self.meta
        self.transform = GeneralizedRCNNTransform(
            min_size=tuple(m["min_size"]),
            max_size=m["max_size"],
            image_mean=m["image_mean"],
            image_std=m["image_std"],
            size_divisible=m["size_divisible"],
        ).eval()

        self.export_h, self.export_w = m["export_shape"]

    # ----------------------------------------------------------------
    def _preprocess(self, images: list[torch.Tensor]):
        """GeneralizedRCNNTransform: resize + normalize + pad."""
        imgs = [im.to(self.device) for im in images]
        image_list, _ = self.transform(imgs, None)

        x = image_list.tensors          # NCHW, уже на GPU
        pad_h, pad_w = x.shape[-2], x.shape[-1]

        # если граф статический — добиваем паддингом до export_shape
        if (pad_h, pad_w) != (self.export_h, self.export_w):
            buf = torch.zeros(
                x.shape[0], 3, self.export_h, self.export_w,
                dtype=x.dtype, device=x.device,
            )
            buf[..., :pad_h, :pad_w] = x
            x = buf

        return x, image_list.image_sizes   # image_sizes — после resize, до pad

    def _run_iobinding(self, x: torch.Tensor):
        """Запускает граф через IOBinding, возвращает три torch-тензора."""
        x = x.contiguous()
        io = self.sess.io_binding()

        # вход — биндим по GPU-указателю, без копии
        io.bind_input(
            name=self.in_name,
            device_type="cuda", device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )

        # выходы — динамический размер (n детекций), ORT аллоцирует сам
        for out in self.sess.get_outputs():
            io.bind_output(out.name, device_type="cuda", device_id=self.device_id)

        self.sess.run_with_iobinding(io)

        boxes, scores, labels = [
            from_dlpack(o.to_dlpack()) for o in io.get_outputs()
        ]
        return boxes, scores, labels

    # ----------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, images: list[torch.Tensor]) -> list[dict]:
        """
        images: list[CHW float32 cuda/cpu], значения 0..1 или 0..255.
        Возвращает list[dict] с ключами boxes (xyxy), scores, labels
        в координатах оригинального изображения.
        """
        orig_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        x, resized_sizes = self._preprocess(images)

        # граф собран под batch=1 — прогоняем по одному
        results = []
        for i in range(x.shape[0]):
            boxes, scores, labels = self._run_iobinding(x[i : i + 1])

            # масштабируем боксы из пространства паддингованного тензора
            # обратно в оригинальный размер изображения
            rh, rw = resized_sizes[i]
            oh, ow = orig_sizes[i]
            scale_x = ow / rw
            scale_y = oh / rh
            boxes = boxes * torch.tensor(
                [scale_x, scale_y, scale_x, scale_y], device=boxes.device
            )
            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        return results

    # ----------------------------------------------------------------
    def warmup(self, n: int = 20):
        dummy = torch.zeros(1, 3, self.export_h, self.export_w, device=self.device)
        for _ in range(n):
            self._run_iobinding(dummy)
        torch.cuda.synchronize()
        print(f"[onnx] warmup {n} iter done")


# --------------------------------------------------------------------
def load_image(path: str, device="cuda") -> torch.Tensor:
    """BGR uint8 -> RGB float32 [0..1] CHW."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device)


def draw_boxes(img_path: str, result: dict, class_names: list[str] | None = None,
               out_path: str | None = None):
    bgr = cv2.imread(img_path)
    boxes  = result["boxes"].cpu().numpy().astype(int)
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()
    for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
        name = class_names[l] if class_names else str(l)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"{name} {s:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out = out_path or img_path.replace(".", "_det.")
    cv2.imwrite(out, bgr)
    print(f"[vis] {out}  ({len(boxes)} boxes)")


# --------------------------------------------------------------------
def benchmark(model: FrcnnOnnxInfer, n: int = 100):
    dummy = [torch.rand(3, 800, 1200, device=model.device)]
    model.warmup(20)
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"latency: mean={times.mean():.1f} ms  "
          f"median={np.median(times):.1f}  "
          f"p95={np.percentile(times, 95):.1f}  "
          f"fps={1000/times.mean():.1f}")


# --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",   required=True, help="путь к frcnn.onnx")
    p.add_argument("--images", nargs="+",     help="изображения для инференса")
    p.add_argument("--bench",  action="store_true", help="замер скорости")
    p.add_argument("--vis",    action="store_true", help="нарисовать боксы")
    p.add_argument("--device", type=int, default=0, help="cuda device id")
    a = p.parse_args()

    model = FrcnnOnnxInfer(a.onnx, device_id=a.device)
    model.warmup()

    if a.bench:
        benchmark(model)

    if a.images:
        for path in a.images:
            img = load_image(path)
            t0 = time.perf_counter()
            res = model([img])[0]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            print(f"{path}: {len(res['boxes'])} boxes  {ms:.1f} ms")
            if a.vis:
                draw_boxes(path, res)
