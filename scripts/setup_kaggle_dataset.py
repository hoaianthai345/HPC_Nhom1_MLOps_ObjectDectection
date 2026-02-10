"""
Download and prepare Traffic Detection dataset from Kaggle
Splits test set into 3 variants for drift testing:
- Part 1: Normal (original)
- Part 2: Brightness modified
- Part 3: Abstract/distorted (blur, noise)
"""
import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from data_version_manager import DataVersionManager


def check_kaggle_setup():
    """
    Check if Kaggle API Token is configured
    """
    api_token = os.getenv('KAGGLE_API_TOKEN')
    if api_token:
        print("✓ Using Kaggle API Token")
        return True
    
    # Not configured
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  Kaggle API Token Not Found                                  ║
╚══════════════════════════════════════════════════════════════════╝

Set your Kaggle API Token:
  export KAGGLE_API_TOKEN="your_api_token"

Get token from: https://www.kaggle.com/settings/account
→ API → Create New Token

Then run this script again.
    """)
    return False


def download_kaggle_dataset(dataset_name: str, download_dir: Path):
    """
    Download dataset from Kaggle using kagglehub
    """
    print(f"\n📥 Downloading dataset from Kaggle...")
    print(f"   Dataset: {dataset_name}")
    
    try:
        import kagglehub
        
        # Download dataset
        path = kagglehub.dataset_download(dataset_name)
        print(f"✓ Downloaded to: {path}")
        
        # kagglehub downloads to cache, need to copy to our directory
        if Path(path).exists():
            print(f"   Copying to: {download_dir}")
            if download_dir.exists():
                shutil.rmtree(download_dir)
            shutil.copytree(path, download_dir, dirs_exist_ok=True)
            print(f"✓ Copied to: {download_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def organize_dataset_structure(raw_dir: Path, output_dir: Path, subset_size: int = None):
    """
    Organize downloaded dataset into YOLO format structure
    
    Args:
        raw_dir: Source directory with dataset
        output_dir: Destination directory
        subset_size: If specified, limit train/val to this many images (test keeps full dataset)
    """
    print(f"\n📁 Organizing dataset structure...")
    if subset_size:
        print(f"   Using subset mode: {subset_size} images for train/val (test keeps full dataset)")
    
    # Expected structure: train/, valid/, test/ with images/ and labels/
    for split in ['train', 'valid', 'test']:
        split_dir = raw_dir / split
        if not split_dir.exists():
            # Try alternative names
            alt_names = [
                'training' if split == 'train' else None,
                'validation' if split == 'valid' else None,
                'testing' if split == 'test' else None
            ]
            for alt_name in alt_names:
                if alt_name and (raw_dir / alt_name).exists():
                    split_dir = raw_dir / alt_name
                    break
        
        if split_dir.exists():
            # Copy to organized structure
            output_split = output_dir / split
            output_split.mkdir(parents=True, exist_ok=True)
            
            # Apply subset only to train and valid, NOT to test
            apply_subset = subset_size and split in ['train', 'valid']
            
            # Copy images (with optional subset limit for train/valid only)
            images_src = split_dir / 'images'
            if images_src.exists():
                if apply_subset:
                    # Copy only subset of images
                    (output_split / 'images').mkdir(parents=True, exist_ok=True)
                    image_files = list(images_src.glob('*.jpg')) + list(images_src.glob('*.png'))
                    for img_file in image_files[:subset_size]:
                        shutil.copy(img_file, output_split / 'images' / img_file.name)
                    print(f"  ✓ Organized {split}/ split ({len(image_files[:subset_size])} images - subset mode)")
                else:
                    shutil.copytree(images_src, output_split / 'images', dirs_exist_ok=True)
                    image_count = len(list((output_split / 'images').glob('*')))
                    print(f"  ✓ Organized {split}/ split ({image_count} images)")
            
            # Copy labels (with optional subset limit for train/valid only)
            labels_src = split_dir / 'labels'
            if labels_src.exists():
                if apply_subset:
                    # Copy only corresponding labels
                    (output_split / 'labels').mkdir(parents=True, exist_ok=True)
                    image_files = list(images_src.glob('*.jpg')) + list(images_src.glob('*.png'))
                    for img_file in image_files[:subset_size]:
                        # Find corresponding label file
                        label_file = labels_src / f"{img_file.stem}.txt"
                        if label_file.exists():
                            shutil.copy(label_file, output_split / 'labels' / label_file.name)
                else:
                    shutil.copytree(labels_src, output_split / 'labels', dirs_exist_ok=True)
    
    # Copy data.yaml if exists
    data_yaml = raw_dir / 'data.yaml'
    if data_yaml.exists():
        shutil.copy(data_yaml, output_dir / 'data.yaml')
        print(f"  ✓ Copied data.yaml")
    else:
        # Create default data.yaml
        create_default_data_yaml(output_dir)
    
    print(f"✓ Dataset organized in: {output_dir}")


def create_default_data_yaml(output_dir: Path):
    """
    Create default data.yaml for YOLO
    """
    # Count classes from labels
    labels_dir = output_dir / 'train' / 'labels'
    classes = set()
    
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)
    
    num_classes = max(classes) + 1 if classes else 1
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Try to infer class names if available
    if num_classes <= 10:
        # Common traffic classes
        common_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                       'person', 'traffic_light', 'stop_sign', 'parking_meter', 'bench']
        class_names = common_names[:num_classes]
    
    data_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"  ✓ Created data.yaml with {num_classes} classes")


def modify_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Modify image brightness
    factor > 1: brighter, factor < 1: darker
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Apply Gaussian blur to simulate abstraction
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    """
    Add Gaussian noise to image
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, size: int = 15) -> np.ndarray:
    """
    Apply motion blur effect
    """
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)


def create_test_variants(original_test_dir: Path, output_base_dir: Path):
    """
    Create 3 variants of test set:
    1. Normal (original)
    2. Brightness modified
    3. Abstract (blur + noise)
    
    Args:
        original_test_dir: Source test directory
        output_base_dir: Destination directory for variants
    """
    print(f"\n🔄 Creating test set variants...")
    print(f"   Using full test dataset for variants (no subset limit)")
    
    variants = {
        'test_normal': {
            'description': 'Original test set without modifications',
            'transform': None
        },
        'test_brightness': {
            'description': 'Test set with brightness modifications',
            'transform': 'brightness'
        },
        'test_abstract': {
            'description': 'Test set with blur and noise (abstract)',
            'transform': 'abstract'
        }
    }
    
    images_dir = original_test_dir / 'images'
    labels_dir = original_test_dir / 'labels'
    
    if not images_dir.exists():
        print(f"✗ Test images not found: {images_dir}")
        return False
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"  Found {len(image_files)} test images")
    
    for variant_name, variant_config in variants.items():
        print(f"\n  Creating {variant_name}...")
        
        variant_dir = output_base_dir / variant_name
        variant_images = variant_dir / 'images'
        variant_labels = variant_dir / 'labels'
        
        variant_images.mkdir(parents=True, exist_ok=True)
        variant_labels.mkdir(parents=True, exist_ok=True)
        
        # Copy labels (same for all variants)
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.txt'):
                shutil.copy(label_file, variant_labels / label_file.name)
        
        # Process images
        for img_file in tqdm(image_files, desc=f"  Processing {variant_name}"):
            img = cv2.imread(str(img_file))
            
            if img is None:
                continue
            
            # Apply transformation
            transform_type = variant_config['transform']
            
            if transform_type == 'brightness':
                # Random brightness factor for each image
                factor = np.random.uniform(0.4, 1.8)  # 40% darker to 80% brighter
                img = modify_brightness(img, factor)
            
            elif transform_type == 'abstract':
                # Apply multiple effects
                # 1. Gaussian blur
                img = apply_gaussian_blur(img, kernel_size=11)
                # 2. Motion blur (random direction)
                if np.random.rand() > 0.5:
                    img = apply_motion_blur(img, size=11)
                # 3. Gaussian noise
                img = add_gaussian_noise(img, mean=0, std=15)
            
            # Save processed image
            output_path = variant_images / img_file.name
            cv2.imwrite(str(output_path), img)
        
        # Create metadata
        metadata = {
            'variant': variant_name,
            'description': variant_config['description'],
            'num_images': len(image_files),
            'transform': transform_type
        }
        
        with open(variant_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"  ✓ Created {variant_name}: {len(image_files)} images")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and setup Kaggle dataset')
    parser.add_argument('--dataset', 
                       default="yusufberksardoan/traffic-detection-project",
                       help='Kaggle dataset name (format: owner/dataset-name)')
    parser.add_argument('--subset', 
                       type=int,
                       default=1000,
                       help='Use only N images for train/val splits (test keeps full dataset) for faster training (e.g., --subset 100)')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     📦 Kaggle Dataset Setup with Test Variants                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Configuration
    KAGGLE_DATASET = args.dataset
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    DOWNLOAD_DIR = DATA_ROOT / "download"
    PROCESSED_DIR = DATA_ROOT / "processed"
    TEST_VARIANTS_DIR = DATA_ROOT / "test_variants"
    
    print(f"\n📊 Dataset: {KAGGLE_DATASET}")
    if args.subset:
        print(f"🔬 Subset Mode: Using {args.subset} images for train/val (test keeps full dataset)")
    
    # Step 1: Check Kaggle API
    if not check_kaggle_setup():
        return
    
    # Step 2: Download dataset
    print("\n" + "="*70)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    if not download_kaggle_dataset(KAGGLE_DATASET, DOWNLOAD_DIR):
        print("✗ Failed to download dataset")
        return
    
    # Step 3: Organize dataset
    print("\n" + "="*70)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    organize_dataset_structure(DOWNLOAD_DIR, PROCESSED_DIR, subset_size=args.subset)
    
    # Step 4: Create test variants
    print("\n" + "="*70)
    TEST_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    
    test_dir = PROCESSED_DIR / 'test'
    if test_dir.exists():
        create_test_variants(test_dir, TEST_VARIANTS_DIR)
    else:
        print(f"✗ Test directory not found: {test_dir}")
    
    # Step 5: Upload to MinIO
    print("\n" + "="*70)
    print("📤 Uploading to MinIO...")
    
    try:
        manager = DataVersionManager(bucket_name="training-data")
        
        # Upload main dataset as v1.0
        print("\n  Uploading training data (v1.0)...")
        version_desc = "Traffic detection dataset from Kaggle (initial version)"
        if args.subset:
            version_desc += f" - SUBSET MODE (train/val: {args.subset} images, test: full dataset)"
        
        success = manager.upload_dataset(
            local_dir=str(PROCESSED_DIR),
            version="v1.0",
            description=version_desc,
            metadata={
                "source": "kaggle",
                "dataset": KAGGLE_DATASET,
                "test_variants": ["normal", "brightness", "abstract"],
                "subset_size": args.subset if args.subset else "full",
                "subset_applies_to": "train_and_val_only" if args.subset else "none"
            }
        )
        
        if success:
            print("  ✓ Training data uploaded to MinIO")
        
    except Exception as e:
        print(f"  ⚠️  MinIO upload skipped: {e}")
        print("  → You can upload later using data_version_manager.py")
    
    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ✅ Dataset Setup Complete!                                      ║
╚══════════════════════════════════════════════════════════════════╝

📁 Directory Structure:
  • Downloaded:      {DOWNLOAD_DIR}
  • Processed:       {PROCESSED_DIR}
  • Test Variants:   {TEST_VARIANTS_DIR}
    ├─ test_normal/       (original)
    ├─ test_brightness/   (brightness modified)
    └─ test_abstract/     (blur + noise)

📊 Dataset Information:
  • Train: {len(list((PROCESSED_DIR / 'train' / 'images').glob('*')))} images
  • Valid: {len(list((PROCESSED_DIR / 'valid' / 'images').glob('*')))} images
  • Test:  {len(list((PROCESSED_DIR / 'test' / 'images').glob('*')))} images

🔬 Test Variants for Drift Detection:
  1. Normal:      Baseline test set
  2. Brightness:  Simulate lighting changes
  3. Abstract:    Simulate blur/noise/quality degradation

🚀 Next Steps:
  1. Train model:
     cd training_pipeline
     python src/train.py
  
  2. Test drift detection:
     python data_validation/drift_analysis.py \\
       --reference_dir data_processed/test/images \\
       --current_dir data_test_variants/test_brightness/images
  
  3. Use in serving pipeline:
     cp -r data_test_variants/test_brightness/images/* \\
       serving_pipeline/production/images/
""")


if __name__ == "__main__":
    main()
