"""
Generate YOLO predictions for all images in dataset.
Saves predictions in YOLO format (.txt files).

Maps COCO classes to dataset classes:
  Dataset: ['bicycle', 'bus', 'car', 'motorbike', 'person']
  COCO:    bicycle=1, bus=5, car=2, motorcycle=3, person=0
"""
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from ultralytics import YOLO

# COCO class ID -> Dataset class ID mapping
# Dataset: 0=bicycle, 1=bus, 2=car, 3=motorbike, 4=person
COCO_TO_DATASET = {
    1: 0,   # COCO bicycle -> dataset bicycle
    5: 1,   # COCO bus -> dataset bus  
    2: 2,   # COCO car -> dataset car
    3: 3,   # COCO motorcycle -> dataset motorbike
    0: 4,   # COCO person -> dataset person
}

# COCO class IDs to filter (only these will be kept)
COCO_CLASSES = list(COCO_TO_DATASET.keys())  # [1, 5, 2, 3, 0]


def generate_predictions(
    model_path: str,
    images_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    classes: Optional[List[int]] = None
):
    """
    Generate predictions for all images, filtering to dataset classes only.
    
    Args:
        model_path: Path to YOLO model weights
        images_dir: Directory containing images
        output_dir: Directory to save prediction .txt files
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        img_size: Image size for inference
        classes: COCO class IDs to filter (default: bicycle, bus, car, motorcycle, person)
    """
    classes = classes or COCO_CLASSES
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Filtering to COCO classes: {classes} (mapped to dataset classes)")
    
    for img_path in tqdm(image_files, desc="Generating predictions"):
        # Run prediction with class filter
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            classes=classes,  # Filter to only these COCO classes
            verbose=False
        )
        
        result = results[0]
        output_file = output_dir / f"{img_path.stem}.txt"
        
        with open(output_file, 'w') as f:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywhn.cpu().numpy()
                coco_classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                for coco_cls, box, conf in zip(coco_classes, boxes, confs):
                    # Map COCO class to dataset class
                    dataset_cls = COCO_TO_DATASET.get(coco_cls)
                    if dataset_cls is not None:
                        x_center, y_center, width, height = box
                        f.write(f"{dataset_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}\n")
    
    print(f"Predictions saved to {output_dir}")
    return len(image_files)


def main():
    """Main function to generate predictions for all splits."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate YOLO predictions (filtered to dataset classes)")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model path")
    parser.add_argument("--data-dir", type=str, default="data_final", help="Data directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to process")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("Class mapping (COCO -> Dataset):")
    print("  COCO 0 (person) -> Dataset 4 (person)")
    print("  COCO 1 (bicycle) -> Dataset 0 (bicycle)")
    print("  COCO 2 (car) -> Dataset 2 (car)")
    print("  COCO 3 (motorcycle) -> Dataset 3 (motorbike)")
    print("  COCO 5 (bus) -> Dataset 1 (bus)")
    
    for split in args.splits:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")
        
        images_dir = data_dir / split / "images"
        output_dir = data_dir / split / "predictions"
        
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
        
        generate_predictions(
            model_path=args.model,
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    
    print("\n All predictions generated!")


if __name__ == "__main__":
    main()
