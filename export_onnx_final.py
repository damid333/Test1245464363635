"""Экспорт Faster R-CNN (torchvision-структура) в два ONNX-графа.

stage1.onnx : image(1,3,H,W), image_hw(2) -> feat0..featN (FPN-уровни для RoI), proposals(M,4)
              backbone + FPN + RPN head + декодирование якорей + top-k + NMS (всё на GPU)
stage2.onnx : feat0..featN, rois(M,4), image_hw(2) -> boxes(D,4), scores(D), labels(D)
              MultiScaleRoIAlign + box head + декодирование + softmax + NMS по классам

Запуск (из корня репо, где установлен fcos_train):
    python export_onnx.py --config runs/exp_X/config.yaml --ckpt runs/exp_X/best.pth \
        --out export/exp_X --canvas 800 1344 [--fp16 --device cuda]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms

# ---------------------------------------------------------------- helpers


def load_model(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Собирает модель тем же build_model, что и train.py, и грузит веса."""
    from fcos_train.config import load_config
    from fcos_train.model import build_model

    cfg = load_config(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = ckpt.get("num_classes", cfg.model.num_classes)
    model = build_model(cfg.model, num_classes)
    state = next((ckpt[k] for k in ("model", "state_dict", "model_state") if k in ckpt), None)
    if state is None:
        raise KeyError(f"Не нашёл веса в чекпойнте, ключи: {list(ckpt.keys())}")
    model.load_state_dict(state)
    model.eval().to(device)

    for attr in ("backbone", "rpn", "roi_heads", "transform"):
        if not hasattr(model, attr):
            raise AttributeError(
                f"У модели нет .{attr}: скрипт рассчитан на Faster R-CNN со структурой torchvision"
            )

    # category_id -> contiguous label (как в обучении)
    mapping = ckpt.get("cat_id_to_contiguous")
    if mapping is None:
        from fcos_train.coco_dataset import CocoDetectionDataset
        from fcos_train.transforms import get_transform

        ds = CocoDetectionDataset(
            cfg.data.train_img_dir, cfg.data.train_ann_file, transforms=get_transform(train=False)
        )
        mapping = ds.cat_id_to_contiguous
    mapping = {int(k): int(v) for k, v in mapping.items()}
    return model, cfg, num_classes, mapping


def decode(deltas, anchors, weights, clip):
    """То же, что torchvision BoxCoder.decode_single, но для произвольной формы (..., 4)."""
    wx, wy, ww, wh = weights
    w = anchors[..., 2] - anchors[..., 0]
    h = anchors[..., 3] - anchors[..., 1]
    cx = anchors[..., 0] + 0.5 * w
    cy = anchors[..., 1] + 0.5 * h
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = torch.clamp(deltas[..., 2] / ww, max=clip)
    dh = torch.clamp(deltas[..., 3] / wh, max=clip)
    pcx = dx * w + cx
    pcy = dy * h + cy
    pw = torch.exp(dw) * w
    ph = torch.exp(dh) * h
    return torch.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], dim=-1)


def clip_to_image(boxes, image_hw):
    """image_hw: float tensor [h, w] — размер отресайзенной картинки без паддинга."""
    x = torch.minimum(boxes[..., 0::2].clamp(min=0), image_hw[1])
    y = torch.minimum(boxes[..., 1::2].clamp(min=0), image_hw[0])
    return torch.stack([x[..., 0], y[..., 0], x[..., 1], y[..., 1]], dim=-1)


def batched_nms_const(boxes, scores, groups, iou_thr, offset: float):
    """batched NMS через сдвиг координат на константу (без boxes.max(),
    чтобы граф не падал на пустом входе)."""
    shifted = boxes + (groups.to(boxes.dtype) * offset)[:, None]
    return nms(shifted, scores, iou_thr)


# ---------------------------------------------------------------- graphs


class Stage1(nn.Module):
    def __init__(self, m, canvas_hw, fp16: bool):
        super().__init__()
        self.backbone = m.backbone
        self.rpn_head = m.rpn.head
        self.fp16 = fp16
        self.roi_names = list(m.roi_heads.box_roi_pool.featmap_names)

        rpn = m.rpn
        self.weights = tuple(float(v) for v in rpn.box_coder.weights)
        self.clip = float(rpn.box_coder.bbox_xform_clip)
        self.post_nms = int(rpn._post_nms_top_n["testing"])
        self.nms_thr = float(rpn.nms_thresh)
        self.score_thr = float(rpn.score_thresh)
        self.min_size = float(rpn.min_size)
        pre_nms = int(rpn._pre_nms_top_n["testing"])

        H, W = canvas_hw
        self.offset = float(max(H, W) + 1)
        dev = next(m.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W, device=dev)
            feats = list(m.backbone(dummy).values())
            anchors = rpn.anchor_generator(ImageList(dummy, [(H, W)]), feats)[0]
        A = rpn.anchor_generator.num_anchors_per_location()[0]
        self.level_shapes = [(A, f.shape[-2], f.shape[-1]) for f in feats]
        counts = [a * h * w for a, h, w in self.level_shapes]
        self.k_per_level = [min(pre_nms, c) for c in counts]
        for i, a in enumerate(torch.split(anchors, counts)):
            self.register_buffer(f"anchors{i}", a.float().clone(), persistent=False)
        levels = torch.cat([torch.full((k,), i, dtype=torch.int64) for i, k in enumerate(self.k_per_level)])
        self.register_buffer("levels", levels.to(dev), persistent=False)

    def forward(self, image, image_hw):
        x = image.half() if self.fp16 else image
        feats = self.backbone(x)
        objs, deltas = self.rpn_head(list(feats.values()))

        boxes, logits = [], []
        for i, (o, d) in enumerate(zip(objs, deltas)):
            A, h, w = self.level_shapes[i]
            # порядок (h, w, A) — как в torchvision concat_box_prediction_layers
            o = o.float().permute(0, 2, 3, 1).reshape(-1)
            d = d.float().reshape(A, 4, h, w).permute(2, 3, 0, 1).reshape(-1, 4)
            top, idx = o.topk(self.k_per_level[i])
            boxes.append(decode(d[idx], getattr(self, f"anchors{i}")[idx], self.weights, self.clip))
            logits.append(top)
        boxes = clip_to_image(torch.cat(boxes), image_hw)
        scores = torch.sigmoid(torch.cat(logits))

        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = torch.where((ws >= self.min_size) & (hs >= self.min_size) & (scores >= self.score_thr))[0]
        boxes, scores, levels = boxes[valid], scores[valid], self.levels[valid]
        keep = batched_nms_const(boxes, scores, levels, self.nms_thr, self.offset)[: self.post_nms]

        roi_feats = [feats[n].float() for n in self.roi_names]
        return (*roi_feats, boxes[keep])


class Stage2(nn.Module):
    def __init__(self, m, canvas_hw, num_classes: int, fp16: bool):
        super().__init__()
        rh = m.roi_heads
        self.pool, self.head, self.pred = rh.box_roi_pool, rh.box_head, rh.box_predictor
        self.names = list(self.pool.featmap_names)
        self.canvas = (int(canvas_hw[0]), int(canvas_hw[1]))
        self.num_classes = num_classes
        self.fp16 = fp16
        self.weights = tuple(float(v) for v in rh.box_coder.weights)
        self.clip = float(rh.box_coder.bbox_xform_clip)
        self.score_thr = float(rh.score_thresh)
        self.nms_thr = float(rh.nms_thresh)
        self.max_dets = int(rh.detections_per_img)
        self.offset = float(max(canvas_hw) + 1)

    def forward(self, *inputs):
        *feats, rois, image_hw = inputs
        pooled = self.pool(OrderedDict(zip(self.names, feats)), [rois], [self.canvas])
        if self.fp16:
            pooled = pooled.half()
        cls, reg = self.pred(self.head(pooled))
        C = self.num_classes
        scores = F.softmax(cls.float(), -1)                                   # (N, C)
        boxes = decode(reg.float().reshape(-1, C, 4), rois[:, None, :], self.weights, self.clip)
        boxes = clip_to_image(boxes, image_hw)                                # (N, C, 4)

        labels = torch.arange(C, device=scores.device).view(1, -1).expand_as(scores)
        boxes, scores, labels = boxes[:, 1:].reshape(-1, 4), scores[:, 1:].reshape(-1), labels[:, 1:].reshape(-1)

        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = torch.where((scores > self.score_thr) & (ws >= 1e-2) & (hs >= 1e-2))[0]
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        keep = batched_nms_const(boxes, scores, labels, self.nms_thr, self.offset)[: self.max_dets]
        return boxes[keep], scores[keep], labels[keep]


# ---------------------------------------------------------------- export


def _onnx_kwargs():
    # в новых torch по умолчанию dynamo-экспортёр; для detection стабильнее старый
    return {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


def export(model, num_classes, mapping, out_dir, canvas_hw, fp16=False, opset=17, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    H, W = canvas_hw
    model = model.to(device).eval()
    s1 = Stage1(model, canvas_hw, fp16).to(device).eval()      # якоря считаются в fp32
    s2 = Stage2(model, canvas_hw, num_classes, fp16).to(device).eval()
    if fp16:
        for mod in (model.backbone, model.rpn.head, model.roi_heads.box_head, model.roi_heads.box_predictor):
            mod.half()

    feat_names = [f"feat{i}" for i in range(len(s1.roi_names))]
    image = torch.randn(1, 3, H, W, device=device)
    hw = torch.tensor([H, W], dtype=torch.float32, device=device)

    with torch.no_grad():
        *feats, props = s1(image, hw)
    p1 = os.path.join(out_dir, "stage1.onnx")
    torch.onnx.export(
        s1, (image, hw), p1, opset_version=opset,
        input_names=["image", "image_hw"], output_names=feat_names + ["proposals"],
        dynamic_axes={"proposals": {0: "M"}}, do_constant_folding=True, **_onnx_kwargs(),
    )

    # для трассировки stage2 нужен непустой набор rois
    if props.shape[0] < 8:
        xy = torch.rand(300, 2, device=device) * torch.tensor([W * 0.8, H * 0.8], device=device)
        props = torch.cat([xy, xy + 40], dim=1)
    p2 = os.path.join(out_dir, "stage2.onnx")
    torch.onnx.export(
        s2, (*feats, props, hw), p2, opset_version=opset,
        input_names=feat_names + ["rois", "image_hw"], output_names=["boxes", "scores", "labels"],
        dynamic_axes={"rois": {0: "M"}, "boxes": {0: "D"}, "scores": {0: "D"}, "labels": {0: "D"}},
        do_constant_folding=True, **_onnx_kwargs(),
    )

    t = model.transform
    meta = {
        "canvas_hw": [H, W],
        "min_size": int(t.min_size[-1]),
        "max_size": int(t.max_size),
        "image_mean": [float(v) for v in t.image_mean],
        "image_std": [float(v) for v in t.image_std],
        "feat_names": feat_names,
        "num_classes": num_classes,
        "contiguous_to_cat_id": {str(v): k for k, v in mapping.items()},
        "fp16": fp16,
        "opset": opset,
        "rpn": {"k_per_level": s1.k_per_level, "post_nms": s1.post_nms, "nms_thr": s1.nms_thr},
        "roi": {"score_thr": s2.score_thr, "nms_thr": s2.nms_thr, "max_dets": s2.max_dets},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[export] {p1}\n[export] {p2}\n[export] {os.path.join(out_dir, 'meta.json')}")
    return p1, p2


def verify_features(model_dir, s1_torch_out, image, hw):
    """Быстрая сверка признаков stage1: torch vs onnxruntime."""
    import onnxruntime as ort

    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(os.path.join(model_dir, "stage1.onnx"), providers=prov)
    outs = sess.run(None, {"image": image.cpu().numpy(), "image_hw": hw.cpu().numpy()})
    for name, a, b in zip([o.name for o in sess.get_outputs()], s1_torch_out, outs):
        a = a.float().cpu().numpy()
        if a.shape == b.shape:
            print(f"[verify] {name:10s} shape={b.shape} max|diff|={abs(a - b).max():.2e}")
        else:
            print(f"[verify] {name:10s} shape torch={a.shape} onnx={b.shape}")
    print(f"[verify] providers: {sess.get_providers()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canvas", type=int, nargs=2, default=[800, 1344], metavar=("H", "W"))
    ap.add_argument("--fp16", action="store_true", help="backbone/FPN/heads в fp16, декодирование и NMS в fp32")
    ap.add_argument("--device", default="cpu", help="для --fp16 используйте cuda")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, _, num_classes, mapping = load_model(args.config, args.ckpt, args.device)
    export(model, num_classes, mapping, args.out, tuple(args.canvas), args.fp16, args.opset, args.device)

    # сверка на случайном входе (признаки должны совпасть до ~1e-4 в fp32)
    model2, *_ = load_model(args.config, args.ckpt, args.device)
    s1 = Stage1(model2, tuple(args.canvas), False).eval()
    if args.fp16:
        print("[verify] fp16: сверка идёт с fp32-моделью, расхождение ~1e-2 — норма")
    H, W = args.canvas
    image = torch.randn(1, 3, H, W, device=args.device)
    hw = torch.tensor([H, W], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        out = s1(image, hw)
    verify_features(args.out, out[:-1], image, hw)


if __name__ == "__main__":
    main()
