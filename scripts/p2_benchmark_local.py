#!/usr/bin/env python3
"""Benchmark latency + throughput local cho mot model YOLO.

Do mean / p95 latency voi batch=1 va throughput cho nhieu batch size.
Cong cu nay khong yeu cau GPU; chay tot tren CPU.

Usage:
    python scripts/p2_benchmark_local.py \
        --model model_artifacts/student_kd_best.pt \
        --val-images data/demo_subset/valid/images \
        --output-dir reports/p2/benchmark \
        --device cpu --batches 1,2,4,8 \
        --warmup 10 --iters 50
"""
from __future__ import annotations
import argparse, csv, glob, json, time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--output-dir", default="reports/p2/benchmark")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batches", default="1,2,4,8",
                   help="Danh sach batch size cach nhau dau phay")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    val_images = Path(args.val_images).resolve()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    batches = [int(b.strip()) for b in args.batches.split(",")]

    from ultralytics import YOLO
    print(f"[bench] model={args.model} device={args.device} imgsz={args.imgsz}")
    m = YOLO(args.model)

    img_pool = sorted(glob.glob(str(val_images / "*")))[:max(32, max(batches))]
    assert img_pool, f"Khong co anh trong {val_images}"

    # ===== Single-image latency (batch=1) =====
    times = []
    for i in range(args.warmup):
        m.predict(img_pool[i % len(img_pool)], imgsz=args.imgsz,
                  conf=0.25, iou=0.45, device=args.device, verbose=False)
    for i in range(args.iters):
        res = m.predict(img_pool[i % len(img_pool)], imgsz=args.imgsz,
                        conf=0.25, iou=0.45, device=args.device, verbose=False)
        times.append(res[0].speed["inference"])
    arr = np.array(times)
    single = {
        "device": args.device,
        "inf_mean_ms": round(float(arr.mean()), 3),
        "inf_p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "inf_p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "fps":         round(1000 / float(arr.mean()), 1),
    }
    print(f"[bench] batch=1 -> {single}")

    # ===== Throughput vs batch =====
    tput = {}
    for bs in batches:
        batch_imgs = (img_pool * ((bs // len(img_pool)) + 1))[:bs]
        for _ in range(args.warmup):
            m.predict(batch_imgs, imgsz=args.imgsz, conf=0.25, iou=0.45,
                      device=args.device, verbose=False)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            m.predict(batch_imgs, imgsz=args.imgsz, conf=0.25, iou=0.45,
                      device=args.device, verbose=False)
        total = time.perf_counter() - t0
        total_imgs = args.iters * bs
        ms_per_img = (total / total_imgs) * 1000
        tput[bs] = {
            "batch_size": bs, "total_s": round(total, 3),
            "total_imgs": total_imgs,
            "ms_per_img": round(ms_per_img, 3),
            "throughput_fps": round(total_imgs / total, 1),
        }
        print(f"[bench] batch={bs} -> {tput[bs]['throughput_fps']} FPS"
              f" ({ms_per_img:.2f} ms/img)")

    out = {
        "model": args.model, "device": args.device, "imgsz": args.imgsz,
        "single": single, "throughput": tput,
    }
    (out_dir / "benchmark_local.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    with open(out_dir / "throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "total_imgs", "total_s", "ms_per_img", "throughput_fps"])
        for bs, v in tput.items():
            w.writerow([v["batch_size"], v["total_imgs"], v["total_s"],
                        v["ms_per_img"], v["throughput_fps"]])

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = list(tput.keys()); ys = [tput[b]["throughput_fps"] for b in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Batch size"); ax.set_ylabel("Throughput (FPS)")
        ax.set_title(f"Throughput vs batch ({args.device})")
        ax.grid(True, alpha=0.3); ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(out_dir / "throughput_curve.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[bench] plot failed: {e}")

    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
