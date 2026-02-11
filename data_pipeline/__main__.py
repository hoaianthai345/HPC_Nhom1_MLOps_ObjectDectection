"""
Unified CLI Entry Point for Data Pipeline

Usage:
    python -m data_pipeline --help
    python -m data_pipeline version list
    python -m data_pipeline version upload --dir data_final --version v1.0
    python -m data_pipeline kaggle download --dataset owner/dataset-name
    python -m data_pipeline migrate --source data_final
    python -m data_pipeline variants create --input test/
"""
import sys
import argparse
from pathlib import Path

from .version_manager import DataVersionManager
from .test_variants import TestVariantCreator
from .kaggle_download import KaggleDownloader
from .subset_creator import DatasetSubsetCreator
from .config import config


def create_parser():
    """Create argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        prog='data_pipeline',
        description='Unified data pipeline for ML training data management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List data versions
  python -m data_pipeline version list
  
  # Upload new version
  python -m data_pipeline version upload --dir data_final --version v1.0
  
  # Download from Kaggle
  python -m data_pipeline kaggle download --dataset owner/dataset-name
  
  # Create subset for demo
  python -m data_pipeline dataset subset --input data/raw --output data/subset --size 1000
  
  # Create test variants
  python -m data_pipeline variants create --input test/ --output test_variants/

For more information, see: docs/DATA_PIPELINE.md
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Version management commands
    version_parser = subparsers.add_parser('version', help='Manage data versions')
    version_sub = version_parser.add_subparsers(dest='version_command')
    
    # version list
    version_sub.add_parser('list', help='List all data versions')
    
    # version info
    info_parser = version_sub.add_parser('info', help='Show version information')
    info_parser.add_argument('--version', required=True, help='Version to show info for')
    
    # version upload
    upload_parser = version_sub.add_parser('upload', help='Upload new data version')
    upload_parser.add_argument('--dir', required=True, help='Directory to upload')
    upload_parser.add_argument('--version', help='Version number (e.g., v1.0)')
    upload_parser.add_argument('--description', default='', help='Version description')
    
    # version download
    download_parser = version_sub.add_parser('download', help='Download a data version')
    download_parser.add_argument('--version', required=True, help='Version to download')
    download_parser.add_argument('--output', required=True, help='Output directory')
    
    # version compare
    compare_parser = version_sub.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('--v1', required=True, help='First version')
    compare_parser.add_argument('--v2', required=True, help='Second version')
    
    # Kaggle commands
    kaggle_parser = subparsers.add_parser('kaggle', help='Download from Kaggle')
    kaggle_sub = kaggle_parser.add_subparsers(dest='kaggle_command', required=True)
    
    # kaggle download
    download_kaggle = kaggle_sub.add_parser('download', help='Download Kaggle dataset')
    download_kaggle.add_argument('--dataset', required=True, help='Dataset name (owner/dataset-name)')
    download_kaggle.add_argument('--output', help='Output directory (default: data/raw/)')
    download_kaggle.add_argument('--organize', action='store_true', help='Organize and show structure')
    download_kaggle.add_argument('--subset', type=int, help='Create subset with N images after download')
    
    # Dataset commands
    dataset_parser = subparsers.add_parser('dataset', help='Dataset operations')
    dataset_sub = dataset_parser.add_subparsers(dest='dataset_command', required=True)
    
    # dataset subset
    subset_parser = dataset_sub.add_parser('subset', help='Create dataset subset')
    subset_parser.add_argument('--input', required=True, help='Input dataset directory')
    subset_parser.add_argument('--output', required=True, help='Output directory')
    subset_parser.add_argument('--size', type=int, default=1000, help='Total number of images (default: 1000)')
    subset_parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio (default: 0.7)')
    subset_parser.add_argument('--valid-ratio', type=float, default=0.2, help='Valid split ratio (default: 0.2)')
    subset_parser.add_argument('--test-ratio', type=float, default=0.1, help='Test split ratio (default: 0.1)')
    
    # dataset analyze
    analyze_parser = dataset_sub.add_parser('analyze', help='Analyze dataset structure')
    analyze_parser.add_argument('--input', required=True, help='Dataset directory to analyze')
    
    # Test variants command
    variants_parser = subparsers.add_parser('variants', help='Create test variants')
    variants_sub = variants_parser.add_subparsers(dest='variants_command')
    
    # variants create
    create_variants = variants_sub.add_parser('create', help='Create test variants')
    create_variants.add_argument('--input', required=True, help='Input test directory')
    create_variants.add_argument('--output', required=True, help='Output directory')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    return parser


def handle_version_command(args):
    """Handle version management commands"""
    manager = DataVersionManager()
    
    if args.version_command == 'list':
        versions = manager.list_versions()
        if versions:
            print("📦 Available data versions:")
            for v in versions:
                info = manager.get_version_info(v)
                desc = info.get('description', 'No description') if info else 'No info'
                created = info.get('created_at', '') if info else ''
                print(f"  • {v:10} ({created}) - {desc}")
        else:
            print("No versions found")
    
    elif args.version_command == 'info':
        info = manager.get_version_info(args.version)
        if info:
            print(f"📋 Version: {args.version}")
            print(f"   Description: {info.get('description', 'N/A')}")
            print(f"   Created: {info.get('created_at', 'N/A')}")
            if 'stats' in info:
                print(f"   Statistics:")
                for split, stats in info['stats'].items():
                    print(f"     {split}: {stats.get('images', 0)} images, {stats.get('labels', 0)} labels")
        else:
            print(f"❌ Version {args.version} not found")
    
    elif args.version_command == 'upload':
        success = manager.upload_dataset(
            local_dir=args.dir,
            version=args.version,
            description=args.description
        )
        return 0 if success else 1
    
    elif args.version_command == 'download':
        success = manager.download_version(args.version, args.output)
        return 0 if success else 1
    
    elif args.version_command == 'compare':
        info1 = manager.get_version_info(args.v1)
        info2 = manager.get_version_info(args.v2)
        
        if info1 and info2:
            print(f"📊 Comparing {args.v1} vs {args.v2}")
            print("\nStatistics:")
            for split in ['train', 'valid', 'test']:
                stats1 = info1.get('stats', {}).get(split, {})
                stats2 = info2.get('stats', {}).get(split, {})
                img1 = stats1.get('images', 0)
                img2 = stats2.get('images', 0)
                diff = img2 - img1
                sign = '+' if diff > 0 else ''
                print(f"  {split:6}: {img1:5} → {img2:5} ({sign}{diff})")
        else:
            print("❌ One or both versions not found")
    
    return 0


def handle_kaggle_command(args):
    """Handle Kaggle download commands"""
    downloader = KaggleDownloader()
    
    if args.kaggle_command == 'download':
        output_dir = Path(args.output) if args.output else None
        
        if args.organize:
            success = downloader.download_and_organize(
                dataset=args.dataset,
                output_dir=output_dir,
                create_subset=bool(args.subset),
                subset_size=args.subset if args.subset else 1000
            )
        else:
            result = downloader.download(args.dataset, output_dir)
            success = result is not None
            
            # Create subset if requested
            if success and args.subset and result:
                print(f"\n📦 Creating subset ({args.subset} images)...")
                from .subset_creator import DatasetSubsetCreator
                
                subset_path = result.parent / f"{result.name}_subset_{args.subset}"
                creator = DatasetSubsetCreator()
                
                success = creator.create_subset(
                    input_dir=result,
                    output_dir=subset_path,
                    subset_size=args.subset
                )
                
                if success:
                    print(f"\n💡 Upload subset to MinIO:")
                    print(f"   python -m data_pipeline version upload --dir {subset_path} --version v1.0")
        
        return 0 if success else 1
    
    return 0


def handle_dataset_command(args):
    """Handle dataset operations"""
    creator = DatasetSubsetCreator()
    
    if args.dataset_command == 'subset':
        distribution = {
            'train': args.train_ratio,
            'valid': args.valid_ratio,
            'test': args.test_ratio
        }
        
        success = creator.create_subset(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            subset_size=args.size,
            distribution=distribution
        )
        return 0 if success else 1
    
    elif args.dataset_command == 'analyze':
        stats = creator.analyze_dataset(Path(args.input))
        
        print(f"\n📊 Dataset Analysis: {args.input}")
        print("=" * 50)
        total = 0
        for split, count in stats.items():
            print(f"  {split:6}: {count:5} images")
            total += count
        print("=" * 50)
        print(f"  Total:  {total:5} images")
        return 0
    
    return 0


def handle_variants_command(args):
    """Handle test variants creation"""
    creator = TestVariantCreator()
    
    if args.variants_command == 'create':
        success = creator.create_variants(
            input_dir=Path(args.input),
            output_dir=Path(args.output)
        )
        return 0 if success else 1
    
    return 0


def handle_config_command(args):
    """Handle configuration commands"""
    print("📝 Data Pipeline Configuration")
    print("="*60)
    print(f"MinIO Endpoint:      {config.minio_endpoint}")
    print(f"Training Bucket:     {config.training_bucket}")
    print(f"Production Bucket:   {config.production_bucket}")
    print(f"Model Exports:       {config.model_exports_bucket}")
    print(f"Project Root:        {config.project_root}")
    print(f"Data Directory:      {config.data_root}")
    
    if args.validate:
        print("\n🔍 Validation:")
        minio_ok = config.validate_minio_config()
        kaggle_ok = config.validate_kaggle_config()
        
        print(f"  MinIO Config:    {'✅ Valid' if minio_ok else '❌ Invalid'}")
        print(f"  Kaggle Config:   {'✅ Valid' if kaggle_ok else '❌ Invalid'}")
    
    return 0


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'version':
            return handle_version_command(args)
        elif args.command == 'kaggle':
            return handle_kaggle_command(args)
        elif args.command == 'dataset':
            return handle_dataset_command(args)
        elif args.command == 'variants':
            return handle_variants_command(args)
        elif args.command == 'config':
            return handle_config_command(args)
        else:
            parser.print_help()
            return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
