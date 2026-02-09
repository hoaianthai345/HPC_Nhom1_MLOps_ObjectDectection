import yaml  # type: ignore
import argparse
import os
from ultralytics import YOLO  # type: ignore
from pathlib import Path

os.environ['AWS_ACCESS_KEY_ID'] = 'minio_admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

try:
    import mlflow  
except ImportError:
    raise ImportError("MLflow is required. Install it with: pip install mlflow")


def load_config(cfg_path: str) -> dict:
    cfg_path_path = Path(cfg_path)
    if not cfg_path_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
    


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Knowledge Distillation Training with MLflow")
    parser.add_argument("config", type=str, help="Path to the training config YAML file")
    parser.add_argument("--teacher-weights", type=str, required=True, help="Path to teacher model weights")
    parser.add_argument("--student-weights", type=str, default=None, help="Path to student model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="runs/mlflow",
                        help="MLflow tracking URI (default: runs/mlflow)")
    parser.add_argument("--mlflow-experiment", type=str, default=None,
                        help="MLflow experiment name (overrides config logging.project)")
    parser.add_argument("--mlflow-run-name", type=str, default=None,
                        help="MLflow run name (overrides config logging.name)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    teacher_weights = args.teacher_weights
    student_path = args.student_weights or "yolo26n.pt"
    data_yaml = args.data


    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    experiment_name = args.mlflow_experiment or cfg.get("logging", {}).get("project", "yolo-distillation")
    mlflow.set_experiment(experiment_name)


    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name


    
    run_name = args.mlflow_run_name or cfg.get("logging", {}).get("name", None)
    if run_name:
        os.environ["MLFLOW_RUN"] = run_name

    from ultralytics import settings as ultra_settings 
    ultra_settings.update({"mlflow": True})

    teacher_model = YOLO(teacher_weights)
    student_model = YOLO(student_path)

    distillation_config = cfg["distillation"]


    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "teacher_weights": teacher_weights,
            "student_weights": student_path,
            "data_yaml": data_yaml,
            "epochs": cfg["training"]["epochs"],
            "imgsz": cfg["training"]["imgsz"],
            "batch": cfg["training"]["batch"],
            "optimizer": cfg["optimization"]["optimizer"],
            "lr0": cfg["optimization"]["lr0"],
            "seed": cfg["training"]["seed"],
        })
        mlflow.log_params({
            f"distill_{k}": v for k, v in distillation_config.items()
        })

    
        results = student_model.train(
            data=data_yaml,

            epochs=cfg["training"]["epochs"],
            imgsz=cfg["training"]["imgsz"],
            batch=cfg["training"]["batch"],
            device=cfg["training"]["device"],
            workers=cfg["training"]["workers"],

            seed=cfg["training"]["seed"],
            deterministic=cfg["training"]["deterministic"],

            optimizer=cfg["optimization"]["optimizer"],
            lr0=cfg["optimization"]["lr0"],
            warmup_epochs=cfg["optimization"]["warmup_epochs"],

            mosaic=cfg["augmentation"]["mosaic"],
            close_mosaic=cfg["augmentation"]["close_mosaic"],

            project=cfg["logging"]["project"],
            name=cfg["logging"]["name"],
            save=cfg["logging"]["save"],
            save_period=cfg["logging"]["save_period"],
            plots=cfg["logging"]["plots"],

            teacher=teacher_model.model,
            distillation_config_loss=distillation_config,
        )

        
        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
        if save_dir and save_dir.exists():
            best_pt = save_dir / "weights" / "best.pt"
            last_pt = save_dir / "weights" / "last.pt"

            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")


            for ext in ("*.png", "*.csv"):
                for f in save_dir.glob(ext):
                    mlflow.log_artifact(str(f), artifact_path="results")

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()