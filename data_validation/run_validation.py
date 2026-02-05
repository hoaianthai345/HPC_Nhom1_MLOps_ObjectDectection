"""
Main entry point for running all data validation and drift analysis.

Usage:
    # Run all analysis (generate predictions + data drift + model performance)
    python -m data_validation.run_validation --all
    
    # Run only data drift analysis (no model predictions needed)
    python -m data_validation.run_validation --data-drift
    
    # Run only model performance analysis (requires predictions)
    python -m data_validation.run_validation --model-performance
    
    # Generate predictions only
    python -m data_validation.run_validation --generate-predictions
    
    # Open HTML reports in browser after run
    python -m data_validation.run_validation --all --open-report
"""
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Data Validation and Drift Analysis for Object Detection"
    )
    
    # Analysis modes
    parser.add_argument("--all", action="store_true", 
                        help="Run all analyses")
    parser.add_argument("--generate-predictions", action="store_true",
                        help="Generate YOLO predictions")
    parser.add_argument("--data-drift", action="store_true",
                        help="Run data drift analysis")
    # (Legacy) Model performance / prediction drift đã bỏ, giữ lại flag để không lỗi nhưng không dùng
    
    # Data settings
    parser.add_argument("--data-dir", type=str, default="data_final",
                        help="Data directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model path for predictions")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory for reports")
    
    # Processing settings
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--splits", type=str, nargs="+", 
                        default=["train", "valid"],
                        help="Splits to process")
    parser.add_argument("--open-report", action="store_true",
                        help="Open HTML reports in default browser after run")
    
    args = parser.parse_args()
    
    # Default to --all if no specific mode selected
    if not any([args.all, args.generate_predictions, args.data_drift]):
        args.all = True
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           🔍 Data Validation & Drift Analysis                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Data Directory:    {args.data_dir:<42} ║
║  Model:             {args.model:<42} ║
║  Output:            {args.output_dir:<42} ║
║  Splits:            {', '.join(args.splits):<42} ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Step 1: Generate predictions
    if args.all or args.generate_predictions:
        print("\n" + "="*60)
        print("STEP 1: Generating YOLO Predictions")
        print("="*60)
        
        from .generate_predictions import generate_predictions
        
        for split in args.splits:
            images_dir = Path(args.data_dir) / split / "images"
            output_dir = Path(args.data_dir) / split / "predictions"
            
            if images_dir.exists():
                print(f"\nProcessing {split}...")
                generate_predictions(
                    model_path=args.model,
                    images_dir=str(images_dir),
                    output_dir=str(output_dir),
                    conf_threshold=args.conf
                )
        
        results["predictions"] = " Generated"
    
    # Step 2: Data Drift Analysis
    if args.all or args.data_drift:
        print("\n" + "="*60)
        print("STEP 2: Data Drift Analysis")
        print("="*60)
        
        from .drift_analysis import run_data_drift_analysis
        
        try:
            drift_result = run_data_drift_analysis(
                data_dir=args.data_dir,
                train_split="train",
                test_split="valid",
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                img_size=args.img_size,
                open_browser=args.open_report
            )
            results["data_drift"] = drift_result
        except Exception as e:
            print(f"❌ Data drift analysis failed: {e}")
            results["data_drift"] = {"error": str(e)}
    
    # Step 3 & 4 (model performance / prediction drift) đã được loại bỏ
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, dict):
            if "error" in value:
                print(f"  {key}:  Failed - {value['error']}")
            elif "passed" in value:
                status = " PASSED" if value["passed"] else "⚠️ FAILED"
                print(f"  {key}: {status}")
                if "report_path" in value:
                    print(f"      Report: {value['report_path']}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n All reports saved to: {args.output_dir}/")
    print("\n Validation complete!")


if __name__ == "__main__":
    main()
