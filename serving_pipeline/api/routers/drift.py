"""
Router for data drift analysis via API (Deepchecks).
"""
from pathlib import Path
from typing import Optional, List, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from data_validation.drift_analysis import run_data_drift_analysis
from ..schemas import DriftResponse


router = APIRouter(prefix="/drift", tags=["Drift"])


@router.get("/data")
def run_data_drift_get(
    format: str = Query(
        "html",
        pattern="^(json|html)$",
        description="Output format: json or html",
    ),
    data_dir: str = Query("data_final", description="Root directory of the dataset"),
    train_split: str = Query("train", description="Train/reference split name"),
    test_split: str = Query("test", description="Test/current split name"),
    output_dir: str = Query("reports", description="Directory to save HTML reports"),
    batch_size: int = Query(32, ge=1, description="Batch size for DataLoader"),
    img_size: int = Query(640, ge=32, description="Image size for Deepchecks pipeline"),
    class_names: Optional[List[str]] = Query(
        default=None,
        description="Optional explicit class names list",
    ),
    max_samples: Optional[int] = Query(
        default=None,
        ge=1,
        description="Optional max number of samples to analyze",
    ),
) -> Any:
    """
    Run data drift analysis.

    - `format=json` → trả JSON `{passed, report_path}`
    - `format=html` → trả nội dung HTML report (hiển thị trên browser)
    """
    try:
        result = run_data_drift_analysis(
            data_dir=data_dir,
            train_split=train_split,
            test_split=test_split,
            output_dir=output_dir,
            batch_size=batch_size,
            img_size=img_size,
            class_names=class_names,
            max_samples=max_samples,
            open_browser=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data drift analysis failed: {exc}") from exc

    # Expose drift scores for Prometheus
    from ..main import DATA_DRIFT_SCORE
    drift_metrics = result.get("drift_metrics") or {}
    for check_name, score in drift_metrics.items():
        DATA_DRIFT_SCORE.labels(check=check_name).set(score)

    report_path: Optional[str] = result.get("report_path")
    if not report_path:
        raise HTTPException(
            status_code=500,
            detail="Data drift analysis did not produce a report",
        )

    if format == "json":
        return DriftResponse(
            passed=bool(result.get("passed", False)),
            report_path=report_path,
        )

    try:
        html_content = Path(report_path).read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read HTML report: {exc}",
        ) from exc

    return HTMLResponse(content=html_content, media_type="text/html")


__all__ = ["router"]
