"""
Analyzes drift between train and validation/test datasets:
- Image Property Drift (brightness, contrast, blurriness)
- Label Drift (class distribution)
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import webbrowser

import numpy as np
from torch.utils.data import DataLoader

from deepchecks.vision import VisionData, Suite
from deepchecks.vision.suites import data_integrity
from deepchecks.vision.checks import ImagePropertyDrift, LabelDrift
from deepchecks.core.check_result import CheckResult
from deepchecks.core.suite import SuiteResult

try:
    from .dataset_loader import load_data_for_deepchecks
except ImportError:
    from dataset_loader import load_data_for_deepchecks

DEFAULT_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']


def _open_report_in_browser(report_path: Path) -> None:
    """Open HTML report in default browser."""
    try:
        webbrowser.open(report_path.resolve().as_uri())
    except Exception:
        pass


class YOLOVisionData(VisionData):
    """Deepchecks VisionData wrapper for YOLO dataset."""
    
    def __init__(self, data_loader: DataLoader, label_map: Optional[Dict[int, str]] = None, **kwargs):
        self.label_map = label_map or {i: name for i, name in enumerate(DEFAULT_CLASSES)}
        super().__init__(data_loader, **kwargs)
    
    def batch_to_images(self, batch: Dict) -> List[np.ndarray]:
        return batch["images"]
    
    def batch_to_labels(self, batch: Dict) -> List[np.ndarray]:
        return batch["labels"]
    
    def get_classes(self, batch_labels: List[np.ndarray]) -> List[List[int]]:
        return [labels[:, 0].astype(int).tolist() if len(labels) > 0 else [] for labels in batch_labels]


def create_vision_data(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> YOLOVisionData:
    """Create Deepchecks VisionData from YOLO dataset."""
    class_names = class_names or DEFAULT_CLASSES
    
    data_loader = load_data_for_deepchecks(
        data_dir=data_dir,
        split=split,
        include_predictions=False,
        batch_size=batch_size,
        img_size=img_size,
        class_names=class_names,
        max_samples=max_samples,
        num_workers=num_workers
    )
    
    return YOLOVisionData(
        data_loader=data_loader,
        label_map={i: name for i, name in enumerate(class_names)},
        task_type='object_detection'
    )


# Property names (sau khi .lower()) từ Deepchecks Image Property Drift để đẩy lên Prometheus
DRIFT_PROPERTIES = (
    "brightness",
    "contrast",
    "rms contrast",
    "mean red relative intensity",
    "mean green relative intensity",
    "mean blue relative intensity",
)


def extract_drift_metrics(suite_result: SuiteResult) -> Dict[str, float]:
    """
    Extract the drift score from the Deepchecks Suite result (check "Image Property Drift").

    Returns only the score for each property: brightness and contrast.

    The value from the check is in the form: { "property_name": {"Drift score": float, "Method": str}, ... }
    Property names in Deepchecks usually have the first letter capitalized (Brightness, Contrast),

    normalization function to lowercase (brightness, contrast) to use as the key.

    Returns:

    Dict with key "brightness", "contrast" (if present in result), value is float [0, 1].
    """
    metrics: Dict[str, float] = {}
    for check_result in suite_result.results:
        if not isinstance(check_result, CheckResult):
            continue
        header = (getattr(check_result, "header", None) or "").strip()
        value = getattr(check_result, "value", None)

        print("[drift_metrics] check header:", repr(header))
        if isinstance(value, dict) and header == "Image Property Drift":
            print("[drift_metrics] Image Property Drift - property names:", list(value.keys()))
            for pname, pdata in value.items():
                if isinstance(pdata, dict):
                    print("[drift_metrics]   ", repr(pname), "->", pdata)
        if not isinstance(value, dict):
            continue
        if header != "Image Property Drift":
            continue
        for prop_name, prop_data in value.items():
            if not isinstance(prop_data, dict):
                continue
            score = prop_data.get("Drift score")
            if score is None:
                continue
            key = (prop_name or "").strip().lower()
            if key in DRIFT_PROPERTIES:
                metrics[key] = float(score)
    return metrics


def run_data_drift_analysis(
    data_dir: str,
    train_split: str = "train",
    test_split: str = "valid",
    output_dir: str = "reports",
    batch_size: int = 32,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    open_browser: bool = False
) -> Dict[str, Any]:
    """Run Data Drift analysis between train and test/valid datasets."""
    print(f"Data Drift Analysis: {train_split} vs {test_split}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {train_split} dataset...")
    train_data = create_vision_data(
        data_dir=data_dir, split=train_split, batch_size=batch_size,
        img_size=img_size, class_names=class_names, max_samples=max_samples
    )
    
    print(f"Loading {test_split} dataset...")
    test_data = create_vision_data(
        data_dir=data_dir, split=test_split, batch_size=batch_size,
        img_size=img_size, class_names=class_names, max_samples=max_samples
    )
    
    print("Running Data Drift checks...")
    drift_suite = Suite(
        "Data Drift Suite",
        ImagePropertyDrift().add_condition_drift_score_less_than(0.15),
        LabelDrift().add_condition_drift_score_less_than(0.15),
    )
    
    result = drift_suite.run(train_dataset=train_data, test_dataset=test_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"data_drift_report_{timestamp}.html"
    result.save_as_html(str(report_path), as_widget=False)
    
    passed = result.passed()
    drift_metrics = extract_drift_metrics(result)
    print(f"Report saved to: {report_path}")
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    if drift_metrics:
        print(f"Drift scores: {drift_metrics}")
    if open_browser:
        _open_report_in_browser(report_path)
    
    return {
        "passed": passed,
        "report_path": str(report_path),
        "result": result,
        "drift_metrics": drift_metrics,
    }


def run_data_integrity_check(
    data_dir: str,
    split: str = "train",
    output_dir: str = "reports",
    batch_size: int = 32,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    open_browser: bool = False
) -> Dict[str, Any]:
    """Run Data Integrity check on a single dataset."""
    print(f"Data Integrity Check: {split}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {split} dataset...")
    data = create_vision_data(
        data_dir=data_dir, split=split, batch_size=batch_size,
        img_size=img_size, class_names=class_names, max_samples=max_samples
    )
    
    print("Running Data Integrity checks...")
    result = data_integrity().run(data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"data_integrity_{split}_{timestamp}.html"
    result.save_as_html(str(report_path), as_widget=False)
    
    passed = result.passed()
    print(f"Report saved to: {report_path}")
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    if open_browser:
        _open_report_in_browser(report_path)
    
    return {"passed": passed, "report_path": str(report_path), "result": result}


def run_prediction_drift_analysis(
    data_dir: str,
    student_predictions_dir: str,
    teacher_predictions_dir: str,
    output_dir: str = "reports",
    batch_size: int = 32,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    open_browser: bool = False,
    num_workers: int = 4
) -> Dict[str, Any]:
    """Run Prediction Drift analysis between student and teacher predictions."""
    print(f"Prediction Drift Analysis: Student vs Teacher")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = class_names or DEFAULT_CLASSES
    
    print(f"Loading student predictions...")
    try:
        from .dataset_loader import YOLODataset, collate_fn
    except ImportError:
        from dataset_loader import YOLODataset, collate_fn
    from torch.utils.data import DataLoader
    import torch
    
    data_dir_path = Path(data_dir)
    
    # Load with student predictions
    student_dataset = YOLODataset(
        images_dir=str(data_dir_path / "images"),
        labels_dir=str(data_dir_path / "labels") if (data_dir_path / "labels").exists() else str(data_dir_path / "images"),
        predictions_dir=student_predictions_dir,
        img_size=img_size,
        class_names=class_names,
        max_samples=max_samples
    )
    
    # Load with teacher predictions
    teacher_dataset = YOLODataset(
        images_dir=str(data_dir_path / "images"),
        labels_dir=str(data_dir_path / "labels") if (data_dir_path / "labels").exists() else str(data_dir_path / "images"),
        predictions_dir=teacher_predictions_dir,
        img_size=img_size,
        class_names=class_names,
        max_samples=max_samples
    )
    
    use_pin_memory = torch.cuda.is_available()
    student_loader = DataLoader(
        student_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=use_pin_memory
    )
    teacher_loader = DataLoader(
        teacher_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=use_pin_memory
    )
    
    student_data = YOLOVisionData(
        data_loader=student_loader,
        label_map={i: name for i, name in enumerate(class_names)},
        task_type='object_detection'
    )
    
    teacher_data = YOLOVisionData(
        data_loader=teacher_loader,
        label_map={i: name for i, name in enumerate(class_names)},
        task_type='object_detection'
    )
    
    print("Running Prediction Drift checks...")
    from deepchecks.vision.checks import PredictionDrift
    
    prediction_drift_suite = Suite(
        "Prediction Drift Suite",
        PredictionDrift().add_condition_drift_score_less_than(0.15),
    )
    
    result = prediction_drift_suite.run(train_dataset=teacher_data, test_dataset=student_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"prediction_drift_report_{timestamp}.html"
    result.save_as_html(str(report_path), as_widget=False)
    
    passed = result.passed()
    print(f"Report saved to: {report_path}")
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    if open_browser:
        _open_report_in_browser(report_path)
    
    return {
        "passed": passed,
        "report_path": str(report_path),
        "result": result,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Data Drift Analysis")
    parser.add_argument("--data-dir", type=str, default="data_final")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="valid")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--open-browser", action="store_true", help="Open report in browser after save")
    
    args = parser.parse_args()
    
    run_data_drift_analysis(
        data_dir=args.data_dir,
        train_split=args.train_split,
        test_split=args.test_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_samples=args.max_samples,
        open_browser=args.open_browser
    )
