# src/fcos_train/inference_v2.py
"""Инференс Faster R-CNN из двух ONNX-графов.

Граф1 (ONNX): backbone + FPN + RPN head              [GPU]
torch:        anchors + decode + NMS proposals       [GPU, torchvision]
torch:        MultiScaleRoIAlign                     [GPU, torchvision]
Граф2 (ONNX): box_head + box_predictor               [GPU]
torch:        postprocess_detections (decode + NMS)  [GPU, torchvision]

Запуск:
    python -m fcos_train.inference_v2 --graphs runs/train2/two_graphs --image img.jpg
    python -m fcos_train.inference_v2 --graphs runs/train2/two_graphs --bench <dir> [--fp16]
"""
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import (
    RegionProposalNetwork, concat_box_prediction_layers)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign


class TwoGraphFrcnn:
    def __init__(self, graph_dir: str, device_id: int = 0, fp16: bool = False,
                 verbose_ort: bool = False):
        gdir = Path(graph_dir)
        with open(gdir / "meta.json") as f:
            self.meta = m = json.load(f)

        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id
        self.static = bool(m.get("static", False))
        self.export_h, self.export_w = m["export_shape"]

        suffix = "_fp16" if fp16 else ""
        cuda_opts = {
            "device_id": device_id,
            # на фиксированном размере можно искать лучший алгоритм один раз
            "cudnn_conv_algo_search": "EXHAUSTIVE" if self.static else "HEURISTIC",
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if verbose_ort:
            so.log_severity_level = 1   # покажет, какие ноды ушли на CPU

        self.sess1 = ort.InferenceSession(
            str(gdir / f"graph1_backbone_rpn{suffix}.onnx"), so, providers)
        self.sess2 = ort.InferenceSession(
            str(gdir / f"graph2_box_head{suffix}.onnx"), so, providers)
        for name, s in [("graph1", self.sess1), ("graph2", self.sess2)]:
            got = s.get_providers()
            print(f"[{name}] providers: {got}")
            if "CUDAExecutionProvider" not in got:
                raise RuntimeError(f"{name}: CUDA EP не поднялся")

        self.fpn_keys = m["fpn_keys"]
        self.n_lvl = len(self.fpn_keys)

        self.transform = GeneralizedRCNNTransform(
            min_size=tuple(m["min_size"]), max_size=m["max_size"],
            image_mean=m["image_mean"], image_std=m["image_std"],
            size_divisible=m["size_divisible"]).eval()

        # RPN без головы: голова в графе 1, здесь только постобработка
        anchor_gen = AnchorGenerator(
            tuple(tuple(s) for s in m["anchor_sizes"]),
            tuple(tuple(r) for r in m["aspect_ratios"]))
        self.rpn = RegionProposalNetwork(
            anchor_gen, nn.Identity(),
            0.7, 0.3, 256, 0.5,                    # используются только в обучении
            pre_nms_top_n={"training": 0, "testing": m["rpn_pre_nms_top_n"]},
            post_nms_top_n={"training": 0, "testing": m["rpn_post_nms_top_n"]},
            nms_thresh=m["rpn_nms_thresh"],
            score_thresh=m["rpn_score_thresh"],
        ).eval()
        self.rpn.min_size = m["rpn_min_size"]

        roi_pool = MultiScaleRoIAlign(
            featmap_names=m["roi_featmap_names"],   # ['0','1','2','3']
            output_size=m["pool_size"],
            sampling_ratio=m["roi_sampling_ratio"])
        self.roi_heads = RoIHeads(
            roi_pool, nn.Identity(), nn.Identity(),
            0.5, 0.5, 512, 0.25,                   # используются только в обучении
            bbox_reg_weights=tuple(m["box_coder_weights"]),
            score_thresh=m["score_thresh"],
            nms_thresh=m["nms_thresh"],
            detections_per_img=m["detections_per_img"]).eval()

        self._anchor_cache: dict = {}

    # ── IOBinding: тензоры не покидают GPU ──────────────────────────
    def _run(self, sess, inputs: dict):
        io = sess.io_binding()
        keep = []   # держим ссылки, чтобы contiguous-копии жили до конца run
        for name, t in inputs.items():
            t = t.contiguous().float()
            keep.append(t)
            io.bind_input(name, "cuda", self.device_id,
                          np.float32, tuple(t.shape), t.data_ptr())
        for o in sess.get_outputs():
            io.bind_output(o.name, "cuda", self.device_id)
        sess.run_with_iobinding(io)
        return [from_dlpack(o.to_dlpack()) for o in io.get_outputs()]

    def _pad_to_export(self, x: torch.Tensor) -> torch.Tensor:
        if not self.static:
            return x
        h, w = x.shape[-2:]
        if h > self.export_h or w > self.export_w:
            raise ValueError(
                f"вход {h}x{w} больше, чем static-граф {self.export_h}x{self.export_w}. "
                f"Переэкспортируй с нужным --height/--width или без --static.")
        if (h, w) == (self.export_h, self.export_w):
            return x
        return F.pad(x, (0, self.export_w - w, 0, self.export_h - h))

    # ── публичный вызов ─────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, images):
        if torch.is_tensor(images):
            images = list(images) if images.dim() == 4 else [images]
        orig_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]

        image_list, _ = self.transform([im.to(self.device) for im in images], None)
        image_list.tensors = self._pad_to_export(image_list.tensors)
        x = image_list.tensors

        # граф 1
        outs = self._run(self.sess1, {"images": x})
        n = self.n_lvl
        feats      = outs[:n]
        objectness = outs[n:2 * n]
        deltas     = outs[2 * n:3 * n]
        feats_dict = OrderedDict(zip(self.fpn_keys, feats))

        # RPN-постобработка (повтор RegionProposalNetwork.forward без head)
        key = tuple(x.shape)
        if key not in self._anchor_cache:
            self._anchor_cache[key] = self.rpn.anchor_generator(image_list, feats)
        anchors = self._anchor_cache[key]

        num_anchors_per_level = [o[0].numel() for o in objectness]
        obj, dlt = concat_box_prediction_layers(objectness, deltas)
        proposals = self.rpn.box_coder.decode(dlt, anchors).view(len(anchors), -1, 4)
        proposals, _ = self.rpn.filter_proposals(
            proposals, obj, image_list.image_sizes, num_anchors_per_level)

        # RoIAlign
        pooled = self.roi_heads.box_roi_pool(
            feats_dict, proposals, image_list.image_sizes)
        if pooled.shape[0] == 0:
            e = torch.zeros(0, 4, device=self.device)
            return [{"boxes": e, "scores": e[:, 0], "labels": e[:, 0].long()}
                    for _ in images]

        # граф 2
        cls_logits, box_reg = self._run(self.sess2, {"pooled": pooled})

        # штатная постобработка torchvision
        boxes, scores, labels = self.roi_heads.postprocess_detections(
            cls_logits.float(), box_reg.float(), proposals, image_list.image_sizes)
        dets = [{"boxes": b, "scores": s, "labels": l}
                for b, s, l in zip(boxes, scores, labels)]
        return self.transform.postprocess(dets, image_list.image_sizes, orig_sizes)

    def warmup(self, n: int = 20, shape=(288, 1536)):
        dummy = [torch.rand(3, *shape, device=self.device)]
        for _ in range(n):
            self(dummy)
        torch.cuda.synchronize()
        print(f"[ok] warmup {n} iter")


# ── CLI ─────────────────────────────────────────────────────────────
def load_image(path, device="cuda") -> torch.Tensor:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device)


def bench(model: TwoGraphFrcnn, image_dir: str, n: int = 100, warmup: int = 20):
    paths = sorted(list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png")))
    paths = [p for p in paths if p.is_file()][:10]
    if not paths:
        raise ValueError(f"нет картинок в {image_dir}")
    imgs = [load_image(p, model.device) for p in paths]

    for i in range(warmup):          # прогрев на реальных кадрах
        model([imgs[i % len(imgs)]])
    torch.cuda.synchronize()

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
    print(f"  fps    = {1000 / times.mean():.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True, help="директория с двумя графами и meta.json")
    p.add_argument("--image", help="одна картинка")
    p.add_argument("--bench", help="директория для замера скорости")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="использовать *_fp16.onnx")
    p.add_argument("--verbose-ort", action="store_true", help="подробный лог ORT (ноды на CPU)")
    a = p.parse_args()

    model = TwoGraphFrcnn(a.graphs, device_id=a.device, fp16=a.fp16,
                          verbose_ort=a.verbose_ort)

    if a.bench:
        bench(model, a.bench, n=a.n)

    if a.image:
        model.warmup(5)
        img = load_image(a.image, model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model([img])[0]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"boxes:  {tuple(res['boxes'].shape)}  {res['boxes'].device}")
        print(f"scores: {res['scores'][:5].tolist()}")
        print(f"labels: {res['labels'][:5].tolist()}")
        print(f"time:   {ms:.1f} ms")
