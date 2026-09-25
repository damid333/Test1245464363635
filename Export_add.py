p.add_argument("--dynamic", action="store_true")
export(a.config, a.ckpt, a.out, a.height, a.width, a.opset, a.dynamic)
def export(cfg_path, ckpt_path, out_path,
           height=800, width=1344, opset=16, dynamic=False):
    ...
    dyn = None
    if dynamic:
        dyn = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes":  {0: "n"},
            "scores": {0: "n"},
            "labels": {0: "n"},
        }

    torch.onnx.export(
        FrcnnGraph(model).eval(), dummy, out_path,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
    )
