#!/usr/bin/env python3
"""ONNX dynamic INT8 quantization + mAP val + CPU latency benchmark.

Cong cu chay hoan toan tren CPU. Dau vao la mot file .onnx FP32 (export tu PyTorch);
script tao ban INT8, do mAP tren tap validation va do latency mean/p95 voi
batch=1 tren 50 anh.

Usage:
    python scripts/p2_quantize_int8.py \
        --onnx model_artifacts/serving_model.onnx \
        --data data/demo_subset/data.yaml \
        --val-images data/demo_subset/valid/images \
        --output-dir reports/p2/int8

Kết quả:
    <output>/serving_model_int8.onnx
    <output>/int8_results.json
"""
from __future__ import annotations
import argparse, json, glob, time
from pathlib import Path

import numpy as np


def benchmark_predict(model_path, images, imgsz, warmup=10, iters=50, device="cpu"):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    for i in range(warmup):
        m.predict(images[i % len(images)], imgsz=imgsz, conf=0.25, iou=0.45,
                  device=device, verbose=False)
    times = []
    for i in range(iters):
        res = m.predict(images[i % len(images)], imgsz=imgsz, conf=0.25, iou=0.45,
                        device=device, verbose=False)
        times.append(res[0].speed["inference"])
    arr = np.array(times)
    return {
        "inf_mean_ms": round(float(arr.mean()), 3),
        "inf_p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "fps":         round(1000 / float(arr.mean()), 1),
    }


def val_map(model_path, data_yaml, imgsz, device="cpu"):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    r = m.val(data=data_yaml, imgsz=imgsz, conf=0.001, iou=0.7,
              split="val", device=device, verbose=False)
    return {
        "map50":     round(float(r.box.map50), 4),
        "map5095":   round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall":    round(float(r.box.mr), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="File .onnx FP32 dau vao")
    p.add_argument("--data", required=True, help="data.yaml cho val mAP")
    p.add_argument("--val-images", required=True, help="Thu muc anh val cho benchmark")
    p.add_argument("--output-dir", default="reports/p2/int8")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    onnx_fp32 = Path(args.onnx).resolve()
    assert onnx_fp32.exists(), f"Khong tim thay {onnx_fp32}"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    onnx_int8 = out_dir / "serving_model_int8.onnx"

    print(f"[quant] input: {onnx_fp32} ({onnx_fp32.stat().st_size/1048576:.2f} MB)")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)
    print(f"[quant] output: {onnx_int8} ({onnx_int8.stat().st_size/1048576:.2f} MB)")

    # Benchmark + val cho ca FP32 va INT8
    images = sorted(glob.glob(str(Path(args.val_images) / "*")))[:50]
    assert images, f"Khong tim thay anh trong {args.val_images}"

    results = {"fp32": {}, "int8": {}}
    for label, path in [("fp32", onnx_fp32), ("int8", onnx_int8)]:
        print(f"\n[{label}] size_mb={path.stat().st_size/1048576:.2f}")
        try:
            mAP = val_map(path, args.data, args.imgsz, device="cpu")
            print(f"[{label}] mAP: {mAP}")
            results[label].update({"size_mb": round(path.stat().st_size/1048576, 2), **mAP})
        except Exception as e:
            print(f"[{label}] val failed: {e}")

        try:
            bench = benchmark_predict(path, images, args.imgsz,
                                       warmup=args.warmup, iters=args.iters, device="cpu")
            print(f"[{label}] latency CPU: {bench}")
            results[label].update(bench)
        except Exception as e:
            print(f"[{label}] benchmark failed: {e}")

    out_json = out_dir / "int8_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[done] saved {out_json}")


if __name__ == "__main__":
    main()
