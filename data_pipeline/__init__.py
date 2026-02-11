"""
Data Pipeline Module

Unified module for managing training data operations:
- Data versioning in MinIO
- Kaggle dataset downloads
- Test variant creation

Usage:
    python -m data_pipeline --help
    python -m data_pipeline version list
    python -m data_pipeline kaggle download --dataset owner/dataset-name
    python -m data_pipeline variants create --input-dir data/test
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"

from .version_manager import DataVersionManager
from .test_variants import TestVariantCreator
from .kaggle_download import KaggleDownloader, download_kaggle_dataset
from .subset_creator import DatasetSubsetCreator, create_dataset_subset
from .config import Config

__all__ = [
    "DataVersionManager",
    "TestVariantCreator",
    "KaggleDownloader",
    "download_kaggle_dataset",
    "DatasetSubsetCreator",
    "create_dataset_subset",
    "Config",
]
