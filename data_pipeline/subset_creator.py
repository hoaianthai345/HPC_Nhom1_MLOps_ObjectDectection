"""
Create subsets of datasets for faster demos and training

This module helps create smaller versions of datasets by sampling
a specified number of images from each split (train/valid/test).
"""
import shutil
import random
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import yaml


class DatasetSubsetCreator:
    """Create subsets of YOLO-format datasets"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize subset creator
        
        Args:
            seed: Random seed for reproducible sampling
        """
        random.seed(seed)
        self.seed = seed
    
    def create_subset(
        self,
        input_dir: Path,
        output_dir: Path,
        subset_size: int,
        distribution: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Create a subset of the dataset
        
        Args:
            input_dir: Input dataset directory (with train/valid/test)
            output_dir: Output directory for subset
            subset_size: Total number of images in subset
            distribution: Distribution of images across splits
                         Default: {'train': 0.7, 'valid': 0.2, 'test': 0.1}
        
        Returns:
            True if successful
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False
        
        # Default distribution
        if distribution is None:
            distribution = {'train': 0.7, 'valid': 0.2, 'test': 0.1}
        
        print(f"\n📊 Creating subset from: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Total size: {subset_size} images")
        print(f"   Distribution: {distribution}")
        
        # Calculate splits
        splits = {}
        remaining = subset_size
        for split, ratio in distribution.items():
            if split == list(distribution.keys())[-1]:
                # Last split gets remaining
                splits[split] = remaining
            else:
                count = int(subset_size * ratio)
                splits[split] = count
                remaining -= count
        
        print(f"\n📦 Split allocation:")
        for split, count in splits.items():
            print(f"   {split:6}: {count:4} images")
        
        # Process each split
        success = True
        total_copied = 0
        
        for split, target_count in splits.items():
            split_dir = input_dir / split
            
            if not split_dir.exists():
                print(f"\n⚠️  Skipping {split} (directory not found)")
                continue
            
            print(f"\n📂 Processing {split}/")
            
            copied = self._copy_split_subset(
                split_dir=split_dir,
                output_dir=output_dir / split,
                target_count=target_count
            )
            
            total_copied += copied
            
            if copied < target_count:
                print(f"   ⚠️  Only {copied}/{target_count} images available")
        
        # Copy or create data.yaml
        self._handle_data_yaml(input_dir, output_dir, splits)
        
        print(f"\n✅ Subset created successfully!")
        print(f"   Total images: {total_copied}")
        print(f"   Location: {output_dir}")
        
        return success
    
    def _copy_split_subset(
        self,
        split_dir: Path,
        output_dir: Path,
        target_count: int
    ) -> int:
        """
        Copy a subset of images and labels from a split
        
        Returns:
            Number of images copied
        """
        # Find all images
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists():
            print(f"   ⚠️  No images directory found")
            return 0
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_dir.glob(ext))
        
        if not image_files:
            print(f"   ⚠️  No images found")
            return 0
        
        # Sample images
        available = len(image_files)
        sample_count = min(target_count, available)
        sampled_images = random.sample(image_files, sample_count)
        
        # Create output directories
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        copied = 0
        for img_path in tqdm(sampled_images, desc=f"   Copying"):
            # Copy image
            shutil.copy2(img_path, output_images / img_path.name)
            
            # Copy label if exists
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name
            
            if label_path.exists():
                shutil.copy2(label_path, output_labels / label_name)
            
            copied += 1
        
        print(f"   ✅ Copied {copied} images")
        return copied
    
    def _handle_data_yaml(
        self,
        input_dir: Path,
        output_dir: Path,
        splits: Dict[str, int]
    ):
        """Copy or create data.yaml file"""
        input_yaml = input_dir / "data.yaml"
        output_yaml = output_dir / "data.yaml"
        
        if input_yaml.exists():
            # Copy and update existing yaml
            with open(input_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Update paths to be relative
            data['train'] = '../train/images'
            data['val'] = '../valid/images'
            data['test'] = '../test/images'
            
            with open(output_yaml, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            print(f"\n📄 Copied data.yaml")
        else:
            # Create basic yaml
            data = {
                'train': '../train/images',
                'val': '../valid/images',
                'test': '../test/images',
                'nc': 1,
                'names': ['object']
            }
            
            with open(output_yaml, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            print(f"\n📄 Created basic data.yaml")
    
    def analyze_dataset(self, dataset_dir: Path) -> Dict:
        """
        Analyze dataset structure and count files
        
        Args:
            dataset_dir: Dataset directory
        
        Returns:
            Dictionary with statistics
        """
        dataset_dir = Path(dataset_dir)
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split / "images"
            
            if split_dir.exists():
                image_count = sum(1 for _ in split_dir.glob('*.[jJ][pP][gG]'))
                image_count += sum(1 for _ in split_dir.glob('*.[pP][nN][gG]'))
                stats[split] = image_count
            else:
                stats[split] = 0
        
        return stats


def create_dataset_subset(
    input_dir: str,
    output_dir: str,
    subset_size: int = 1000,
    distribution: Optional[Dict[str, float]] = None
) -> bool:
    """
    Convenience function to create dataset subset
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory
        subset_size: Total number of images
        distribution: Optional custom distribution
    
    Returns:
        True if successful
    """
    creator = DatasetSubsetCreator()
    return creator.create_subset(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        subset_size=subset_size,
        distribution=distribution
    )
