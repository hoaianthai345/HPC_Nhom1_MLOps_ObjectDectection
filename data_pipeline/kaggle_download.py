"""
Download datasets from Kaggle using kagglehub API

Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables set,
or ~/.kaggle/kaggle.json file with credentials.
"""
import os
import shutil
from pathlib import Path
from typing import Optional
import kagglehub
from .config import config
from .subset_creator import DatasetSubsetCreator


class KaggleDownloader:
    """Download and manage Kaggle datasets"""
    
    def __init__(self):
        """Initialize Kaggle downloader"""
        self.validate_credentials()
    
    def validate_credentials(self) -> bool:
        """Check if Kaggle credentials are configured"""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_file = kaggle_json.exists()
        has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        
        if not (has_file or has_env):
            print("⚠️  Warning: Kaggle credentials not found")
            print("   Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            print("   Or create ~/.kaggle/kaggle.json")
            return False
        return True
    
    def download(self, dataset: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Download dataset from Kaggle
        
        Args:
            dataset: Kaggle dataset in format "owner/dataset-name"
            output_dir: Optional output directory. If None, uses kagglehub cache
        
        Returns:
            Path to downloaded dataset directory
        """
        try:
            print(f"📥 Downloading Kaggle dataset: {dataset}")
            print("   This may take a few minutes...")
            
            # Download using kagglehub
            download_path = kagglehub.dataset_download(dataset)
            download_path = Path(download_path)
            
            print(f"✅ Downloaded to: {download_path}")
            
            # If output_dir specified, copy files there
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"📂 Copying to: {output_dir}")
                
                # Copy all files from download path to output dir
                for item in download_path.iterdir():
                    dest = output_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                
                print(f"✅ Copied to: {output_dir}")
                return output_dir
            
            return download_path
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None
    
    def download_and_organize(
        self,
        dataset: str,
        output_dir: Optional[Path] = None,
        structure: str = "auto",
        create_subset: bool = False,
        subset_size: int = 1000
    ) -> bool:
        """
        Download dataset and organize it
        
        Args:
            dataset: Kaggle dataset name
            output_dir: Output directory (default: data/raw/)
            structure: Directory structure ('auto', 'flat', 'yolo')
            create_subset: Whether to create a subset after download
            subset_size: Size of subset if create_subset=True
        
        Returns:
            True if successful
        """
        if output_dir is None:
            output_dir = config.data_root / "raw"
        
        download_path = self.download(dataset, output_dir)
        
        if download_path:
            print(f"\n📊 Dataset structure:")
            self._print_structure(download_path, max_depth=2)
            
            # Create subset if requested
            if create_subset:
                print(f"\n📦 Creating subset ({subset_size} images)...")
                subset_path = download_path.parent / f"{download_path.name}_subset_{subset_size}"
                
                creator = DatasetSubsetCreator()
                success = creator.create_subset(
                    input_dir=download_path,
                    output_dir=subset_path,
                    subset_size=subset_size
                )
                
                if success:
                    print(f"\n✅ Subset created at: {subset_path}")
                    print(f"\n💡 Next steps:")
                    print(f"   Upload subset to MinIO:")
                    print(f"   python -m data_pipeline version upload --dir {subset_path} --version v1.0")
                    return True
            else:
                print(f"\n💡 Next steps:")
                print(f"   1. Review structure: ls -R {download_path}")
                print(f"   2. Create subset for demo:")
                print(f"      python -m data_pipeline dataset subset --input {download_path} --output {download_path}_subset --size 1000")
                print(f"   3. Upload to MinIO:")
                print(f"      python -m data_pipeline version upload --dir {download_path} --version v1.0")
            
            return True
        
        return False
    
    def _print_structure(self, path: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0):
        """Print directory structure"""
        if current_depth >= max_depth:
            return
        
        try:
            items = sorted(path.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Print directories first
            for i, item in enumerate(dirs):
                is_last = i == len(dirs) - 1 and len(files) == 0
                print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
                
                if current_depth < max_depth - 1:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    self._print_structure(item, new_prefix, max_depth, current_depth + 1)
            
            # Print files (limited)
            file_count = len(files)
            display_files = files[:3]  # Show first 3 files
            
            for i, item in enumerate(display_files):
                is_last = i == len(display_files) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
            
            if file_count > 3:
                print(f"{prefix}    ... and {file_count - 3} more files")
                
        except PermissionError:
            pass


def download_kaggle_dataset(dataset: str, output_dir: Optional[str] = None) -> Optional[Path]:
    """
    Convenience function to download Kaggle dataset
    
    Args:
        dataset: Kaggle dataset name (owner/dataset-name)
        output_dir: Optional output directory
    
    Returns:
        Path to downloaded dataset
    """
    downloader = KaggleDownloader()
    output_path = Path(output_dir) if output_dir else None
    return downloader.download(dataset, output_path)
