# src/fcos_train/infer_two_graphs.py
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.dlpack import from_dlpack
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign


class TwoGraphFrcnn:
    """
    Граф1 (ONNX): backbone + FPN + RPN  → фичемапы + proposals  [GPU]
    torch:         MultiScaleRoIAlign                              [GPU, torchvision.ops]
    Граф2 (ONNX): box_head + box_predictor → logits + regression  [GPU]
    torch:         decode + NMS                                    [GPU, torchvision.ops]
    """

    def __init__(self, graph_dir: str, device_id: int = 0):
        with open(Path(graph_dir) / "meta.json") as f:
            self.meta = json.load(f)
        m = self.meta
        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id

        cuda_opts = {"device_id": device_id,
                     "cudnn_conv_algo_search": "HEURISTIC"}
        providers = [("CUDAExecutionProvider", cuda_opts),
                     "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess1 = ort.InferenceSession(
            str(Path(graph_dir) / "graph1_backbone_rpn.onnx"), so, providers)
        self.sess2 = ort.InferenceSession(
            str(Path(graph_dir) / "graph2_box_head.onnx"), so, providers)

        for name, sess in [("graph1", self.sess1), ("graph2", self.sess2)]:
            got = sess.get_providers()
            print(f"[{name}] providers: {got}")
            if "CUDAExecutionProvider" not in got:
                raise RuntimeError(f"{name}: CUDA EP не поднялся")

        self.fpn_keys = m["fpn_keys"]
        self.out_ch   = m["out_channels"]
        self.pool_sz  = m["pool_size"]

        self.transform = GeneralizedRCNNTransform(
            min_size=tuple(m["min_size"]), max_size=m["max_size"],
            image_mean=m["image_mean"], image_std=m["image_std"],
            size_divisible=m["size_divisible"],
        ).eval()

        # RoiAlign на GPU через torchvision.ops — без ONNX
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=self.fpn_keys,
            output_size=self.pool_sz,
            sampling_ratio=2,
        )

        self.score_thresh      = m["score_thresh"]
        self.nms_thresh        = m["nms_thresh"]
        self.detections_per_img = m["detections_per_img"]

    # ── IOBinding helper ────────────────────────────────────────────
    def _run(self, sess, inputs: dict[str, torch.Tensor]):
        io = sess.io_binding()
        for name, t in inputs.items():
            t = t.contiguous()
            io.bind_input(name, "cuda", self.device_id,
                          np.float32, tuple(t.shape), t.data_ptr())
        for o in sess.get_outputs():
            io.bind_output(o.name, "cuda", self.device_id)
        sess.run_with_iobinding(io)
        return [from_dlpack(o.to_dlpack()) for o in io.get_outputs()]

    # ── decode + NMS (torch, GPU) ───────────────────────────────────
    def _postprocess(self, cls_logits, box_reg, proposals, image_size):
        import torch.nn.functional as F
        from torchvision.ops import batched_nms, clip_boxes_to_image

        scores = F.softmax(cls_logits, dim=-1)      # [N, num_classes]
        num_classes = scores.shape[-1]

        # раскладываем боксы: [N, num_classes*4] -> [N, num_classes, 4]
        boxes = box_reg.reshape(-1, num_classes, 4)

        # декодируем относительно proposals
        wx, wy, ww, wh = 10.0, 10.0, 5.0, 5.0
        px = (proposals[:, 0] + proposals[:, 2]) / 2
        py = (proposals[:, 1] + proposals[:, 3]) / 2
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]

        dx = boxes[:, :, 0] / wx
        dy = boxes[:, :, 1] / wy
        dw = boxes[:, :, 2] / ww
        dh = boxes[:, :, 3] / wh

        dw = torch.clamp(dw, max=np.log(1000.0 / 16))
        dh = torch.clamp(dh, max=np.log(1000.0 / 16))

        pred_cx = dx * pw[:, None] + px[:, None]
        pred_cy = dy * ph[:, None] + py[:, None]
        pred_w  = torch.exp(dw) * pw[:, None]
        pred_h  = torch.exp(dh) * ph[:, None]

        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [N, C, 4]

        all_boxes, all_scores, all_labels = [], [], []
        for cls in range(1, num_classes):    # 0 = background
            s = scores[:, cls]
            b = pred_boxes[:, cls]
            b = clip_boxes_to_image(b, image_size)
            keep = s > self.score_thresh
            s, b = s[keep], b[keep]
            keep2 = batched_nms(b, s,
                                torch.full_like(s, cls, dtype=torch.long),
                                self.nms_thresh)
            all_boxes.append(b[keep2])
            all_scores.append(s[keep2])
            all_labels.append(torch.full((keep2.numel(),), cls,
                                         dtype=torch.long, device=s.device))

        boxes_out  = torch.cat(all_boxes)
        scores_out = torch.cat(all_scores)
        labels_out = torch.cat(all_labels)

        # топ-N
        if len(scores_out) > self.detections_per_img:
            topk = scores_out.topk(self.detections_per_img).indices
            boxes_out  = boxes_out[topk]
            scores_out = scores_out[topk]
            labels_out = labels_out[topk]

        return {"boxes": boxes_out, "scores": scores_out, "labels": labels_out}

    # ── публичный вызов ─────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, images):
        if torch.is_tensor(images):
            images = list(images)
        orig_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        image_list, _ = self.transform(
            [im.to(self.device) for im in images], None)
        x = image_list.tensors

        # ── граф 1 ──────────────────────────────────────────────────
        outs1 = self._run(self.sess1, {"images": x})
        feats_list = outs1[:len(self.fpn_keys)]
        proposals  = outs1[-1]   # [N_prop, 4]

        # ── RoiAlign (torch, GPU) ────────────────────────────────────
        feats_dict = OrderedDict(zip(self.fpn_keys, feats_list))
        pooled = self.roi_pool(
            feats_dict,
            [proposals],
            image_list.image_sizes,
        )   # [N_prop, C, pool_sz, pool_sz]

        # ── граф 2 ──────────────────────────────────────────────────
        cls_logits, box_reg = self._run(self.sess2, {"pooled": pooled})

        # ── decode + NMS (torch, GPU) ────────────────────────────────
        results = []
        for i in range(len(images)):
            det = self._postprocess(
                cls_logits, box_reg, proposals,
                image_list.image_sizes[i],
            )
            # скейл к оригиналу
            rh, rw = image_list.image_sizes[i]
            oh, ow = orig_sizes[i]
            det["boxes"] *= torch.tensor(
                [ow/rw, oh/rh, ow/rw, oh/rh], device=self.device)
            results.append(det)
        return results

    def warmup(self, n=20):
        dummy = [torch.rand(3, 600, 800, device=self.device)]
        for _ in range(n):
            self(dummy)
        torch.cuda.synchronize()
        print(f"[ok] warmup {n} iter")


# ── CLI ─────────────────────────────────────────────────────────────
def load_image(path, device="cuda"):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device)


def bench(model, image_dir, n=100):
    paths = list(Path(image_dir).glob("*.jpg"))[:10]
    if not paths:
        paths = list(Path(image_dir).glob("*.png"))[:10]
    imgs = [load_image(p, model.device) for p in paths]
    model.warmup(20)
    times = []
    for i in range(n):
        img = imgs[i % len(imgs)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model([img])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"ONNX two-graph Faster R-CNN  (n={n})")
    print(f"  mean   = {times.mean():.1f} ms")
    print(f"  median = {np.median(times):.1f} ms")
    print(f"  p95    = {np.percentile(times, 95):.1f} ms")
    print(f"  fps    = {1000/times.mean():.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graphs",  required=True, help="директория с двумя графами")
    p.add_argument("--image",   help="одна картинка")
    p.add_argument("--bench",   help="директория для замера скорости")
    p.add_argument("--device",  type=int, default=0)
    a = p.parse_args()

    model = TwoGraphFrcnn(a.graphs, device_id=a.device)
    model.warmup()

    if a.bench:
        bench(model, a.bench)

    if a.image:
        img = load_image(a.image, model.device)
        t0 = time.perf_counter()
        res = model([img])[0]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"boxes: {res['boxes'].shape}  {res['boxes'].device}")
        print(f"scores: {res['scores'].shape}")
        print(f"labels: {res['labels'].shape}")
        print(f"time: {ms:.1f} ms")
