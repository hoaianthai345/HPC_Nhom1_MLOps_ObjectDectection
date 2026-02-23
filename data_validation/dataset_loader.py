"""
Dataset loader for YOLO format data compatible with Deepchecks.

Label format (YOLO): class_id x_center y_center width height (normalized 0-1)
Output format (Deepchecks): [class_id, x, y, w, h] in pixel coordinates
Prediction format (Deepchecks): [x, y, w, h, confidence, class_id] in pixel coordinates
"""
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class YOLODataset(Dataset):
    """PyTorch Dataset for YOLO format data."""
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        predictions_dir: Optional[str] = None,
        img_size: int = 640,
        class_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing ground truth labels (.txt)
            predictions_dir: Directory containing predictions (.txt), optional
            img_size: Target image size (uses letterbox to preserve aspect ratio)
            class_names: List of class names
            max_samples: Maximum number of samples to load (None = all)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.predictions_dir = Path(predictions_dir) if predictions_dir else None
        self.img_size = img_size
        self.class_names = class_names or ['bicycle', 'bus', 'car', 'motorbike', 'person']
        
        # Get image files
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) + 
            list(self.images_dir.glob("*.png"))
        )
        
        # Validate dataset
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Limit samples if specified
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index."""
        img_path = self.image_files[idx]
        
        # Load and letterbox resize image (preserve aspect ratio)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_np, scale, pad = self._letterbox(np.array(image), self.img_size)
        
        # Load labels with letterbox transform
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path, orig_w, orig_h, scale, pad)
        
        # Load predictions if available
        predictions = None
        if self.predictions_dir:
            pred_path = self.predictions_dir / f"{img_path.stem}.txt"
            predictions = self._load_predictions(pred_path, orig_w, orig_h, scale, pad)
        
        result = {
            "image": image_np,
            "labels": labels,
            "image_path": str(img_path),
        }
        if predictions is not None:
            result["predictions"] = predictions
        
        return result
    
    def _letterbox(self, img: np.ndarray, target_size: int) -> tuple:
        """
        Resize image with letterbox (preserve aspect ratio with padding).
        
        Returns:
            (resized_image, scale, (pad_w, pad_h))
        """
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        
        # Pad to target size
        pad_w, pad_h = (target_size - new_w) // 2, (target_size - new_h) // 2
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return canvas, scale, (pad_w, pad_h)
    
    def _load_labels(self, label_path: Path, orig_w: int, orig_h: int, 
                      scale: float, pad: tuple) -> np.ndarray:
        """
        Load labels from YOLO format and transform to letterboxed coordinates.
        
        Returns:
            [N, 5] array with format [class_id, x, y, w, h] in pixel coordinates
        """
        labels = []
        pad_w, pad_h = pad
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Denormalize to original size, then apply letterbox transform
                        x_center = float(parts[1]) * orig_w * scale + pad_w
                        y_center = float(parts[2]) * orig_h * scale + pad_h
                        w = float(parts[3]) * orig_w * scale
                        h = float(parts[4]) * orig_h * scale
                        
                        # Convert center to top-left corner
                        x = x_center - w / 2
                        y = y_center - h / 2
                        
                        # Clip coordinates to valid range (non-negative)
                        x = max(0.0, x)
                        y = max(0.0, y)
                        # Clip width and height to stay within image bounds
                        w = min(w, self.img_size - x)
                        h = min(h, self.img_size - y)
                        
                        labels.append([class_id, x, y, w, h])
        
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
    
    def _load_predictions(self, pred_path: Path, orig_w: int, orig_h: int,
                          scale: float, pad: tuple) -> np.ndarray:
        """
        Load predictions from YOLO format and transform to letterboxed coordinates.
        
        Returns:
            [N, 6] array with format [x, y, w, h, confidence, class_id]
        """
        predictions = []
        pad_w, pad_h = pad
        missing_conf_warned = False
        
        if pred_path.exists():
            with open(pred_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * orig_w * scale + pad_w
                        y_center = float(parts[2]) * orig_h * scale + pad_h
                        w = float(parts[3]) * orig_w * scale
                        h = float(parts[4]) * orig_h * scale
                        
                        if len(parts) > 5:
                            confidence = float(parts[5])
                        else:
                            confidence = 1.0
                            if not missing_conf_warned:
                                warnings.warn(f"Missing confidence in {pred_path}, defaulting to 1.0")
                                missing_conf_warned = True
                        
                        x = x_center - w / 2
                        y = y_center - h / 2
                        
                        # Clip coordinates to valid range (non-negative)
                        x = max(0.0, x)
                        y = max(0.0, y)
                        # Clip width and height to stay within image bounds
                        w = min(w, self.img_size - x)
                        h = min(h, self.img_size - y)
                        
                        predictions.append([x, y, w, h, confidence, class_id])
        
        return np.array(predictions, dtype=np.float32) if predictions else np.zeros((0, 6), dtype=np.float32)


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function - returns lists for Deepchecks."""
    result = {
        "images": [item["image"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_identifiers": [item["image_path"] for item in batch],
    }
    if batch and "predictions" in batch[0]:
        result["predictions"] = [item["predictions"] for item in batch]
    return result


def load_data_for_deepchecks(
    data_dir: str,
    split: str = "train",
    include_predictions: bool = False,
    batch_size: int = 32,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Load data for Deepchecks analysis.
    
    Args:
        data_dir: Root data directory (containing train/valid/test subdirs)
        split: Dataset split (train, valid, test)
        include_predictions: Whether to include predictions
        batch_size: Batch size
        img_size: Image size
        class_names: List of class names
        max_samples: Maximum samples to load (None = all)
        num_workers: Number of workers
        
    Returns:
        DataLoader
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"
    predictions_dir = None
    
    if include_predictions:
        pred_dir = data_dir / split / "predictions"
        if pred_dir.exists():
            predictions_dir = str(pred_dir)
        else:
            warnings.warn(f"Predictions directory not found: {pred_dir}")
    
    dataset = YOLODataset(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        predictions_dir=predictions_dir,
        img_size=img_size,
        class_names=class_names,
        max_samples=max_samples
    )
    
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
