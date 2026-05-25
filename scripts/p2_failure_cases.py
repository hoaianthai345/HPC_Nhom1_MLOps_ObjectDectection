#!/usr/bin/env python3
"""Tim va render N anh validation co recall thap nhat (failure cases).

Quet mot so anh validation, chay du doan voi mot model YOLO, so sanh voi
ground truth tai nguong IoU cho truoc, va render N anh worst-case (3 cot:
Input | Ground Truth | Prediction).

Usage:
    python scripts/p2_failure_cases.py \
        --model model_artifacts/student_kd_best.pt \
        --val-images data/demo_subset/valid/images \
        --val-labels data/demo_subset/valid/labels \
        --output-dir reports/p2/failure_cases \
        --n 5 --iou 0.5 --device cpu --sample 150
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_CLASSES = ["bicycle", "bus", "car", "motorbike", "person"]


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    p = Path(label_path)
    if not p.exists(): return boxes
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        cid, xc, yc, w, h = int(parts[0]), *(float(x) for x in parts[1:5])
        x1 = (xc - w/2) * img_w; y1 = (yc - h/2) * img_h
        x2 = (xc + w/2) * img_w; y2 = (yc + h/2) * img_h
        boxes.append((cid, x1, y1, x2, y2))
    return boxes


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw_boxes(img, boxes, color, names):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for cid, x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1, max(0, y1 - 12)),
               names[cid] if cid < len(names) else str(cid), fill=color)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-labels", default=None, help="Mac dinh: <val-images>/../labels")
    p.add_argument("--output-dir", default="reports/p2/failure_cases")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sample", type=int, default=150,
                   help="So anh val toi da de quet (de tiet kiem CPU)")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    args = p.parse_args()

    val_images = Path(args.val_images).resolve()
    val_labels = Path(args.val_labels).resolve() if args.val_labels else val_images.parent / "labels"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    class_names = args.classes.split(",")

    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    m = YOLO(args.model)

    imgs = sorted(glob.glob(str(val_images / "*")))[:args.sample]
    assert imgs, f"Khong co anh trong {val_images}"
    print(f"[failure] quet {len(imgs)} anh, lay top-{args.n} worst (IoU >= {args.iou})")

    scored = []
    for ip in imgs:
        lp = val_labels / (Path(ip).stem + ".txt")
        img = Image.open(ip); W, H = img.size
        gt = load_yolo_labels(lp, W, H)
        if not gt: continue
        res = m.predict(ip, imgsz=args.imgsz, conf=0.25, iou=0.45,
                        device=args.device, verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cid = int(b.cls.item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append((cid, x1, y1, x2, y2))
        matched = 0
        for g in gt:
            best = 0.0
            for pr in preds:
                if pr[0] != g[0]: continue
                best = max(best, iou(pr[1:], g[1:]))
            if best >= args.iou: matched += 1
        recall_img = matched / len(gt)
        scored.append({
            "img": ip, "gt": gt, "preds": preds,
            "recall": recall_img, "missed": len(gt) - matched,
            "fp": len(preds) - matched, "n_gt": len(gt),
        })

    scored.sort(key=lambda s: (s["recall"], -s["missed"]))
    worst = scored[: args.n]
    print(f"[failure] tim duoc {len(worst)} anh worst tu {len(scored)} anh co GT")

    meta = []
    for i, s in enumerate(worst, 1):
        img = Image.open(s["img"]).convert("RGB")
        img_gt = draw_boxes(img, s["gt"], "lime", class_names)
        img_pr = draw_boxes(img, s["preds"], "red", class_names)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img);    ax[0].set_title("Input");           ax[0].axis("off")
        ax[1].imshow(img_gt); ax[1].set_title(f"GT ({s['n_gt']} obj)"); ax[1].axis("off")
        ax[2].imshow(img_pr)
        ax[2].set_title(f"Pred (R={s['recall']:.2f}, miss={s['missed']}, FP={s['fp']})")
        ax[2].axis("off")
        out_png = out_dir / f"failure_{i}.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
        meta.append({"index": i, "image": Path(s["img"]).name,
                     "recall": s["recall"], "n_gt": s["n_gt"],
                     "missed": s["missed"], "fp": s["fp"],
                     "render": out_png.name})
        print(f"  {i}. {Path(s['img']).name} recall={s['recall']:.2f} missed={s['missed']}/{s['n_gt']}")

    (out_dir / "failure_cases.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[done] saved {out_dir}")


if __name__ == "__main__":
    main()
