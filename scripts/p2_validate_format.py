#!/usr/bin/env python3
"""Validate mAP cho mot model file (.pt / .onnx / .engine) tren tap validation.

Usage:
    python scripts/p2_validate_format.py \
        --model model_artifacts/serving_model.pt \
        --data data/demo_subset/data.yaml \
        --imgsz 640 --conf 0.001 --iou 0.7 --device cpu \
        --output reports/p2/val_pt.json

Ket qua xuat ra mot file JSON gom mAP50, mAP50-95, Precision, Recall, size MB.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Duong dan .pt/.onnx/.engine")
    p.add_argument("--data", required=True, help="data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cpu", help="cpu hoac 0/1/...")
    p.add_argument("--split", default="val")
    p.add_argument("--output", default=None, help="File JSON ket qua")
    args = p.parse_args()

    model_path = Path(args.model).resolve()
    assert model_path.exists(), f"Khong tim thay model: {model_path}"

    from ultralytics import YOLO

    print(f"[validate] model={model_path} ({model_path.stat().st_size/1048576:.2f} MB)")
    print(f"[validate] data={args.data} imgsz={args.imgsz} conf={args.conf} iou={args.iou} device={args.device}")

    m = YOLO(str(model_path), task="detect")
    r = m.val(
        data=args.data, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        split=args.split, device=args.device, verbose=False,
    )

    result = {
        "model": str(model_path),
        "format": model_path.suffix.lstrip("."),
        "size_mb": round(model_path.stat().st_size / 1048576, 2),
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
        "device": args.device, "split": args.split,
        "map50":     round(float(r.box.map50), 4),
        "map5095":   round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall":    round(float(r.box.mr), 4),
    }
    # Per-class mAP nếu có
    try:
        per_class = {f"class_{i}": round(float(v), 4) for i, v in enumerate(r.box.maps.tolist())}
        result["per_class_map5095"] = per_class
    except Exception:
        pass

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[validate] saved {out}")


if __name__ == "__main__":
    main()
