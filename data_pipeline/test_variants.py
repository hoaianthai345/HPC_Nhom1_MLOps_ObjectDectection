"""
Test Variant Creator for Drift Testing

Creates different variants of test sets to simulate data drift:
- Brightness changes
- Contrast changes
- Noise addition
- Blur effects

Refactored for consistency with data_pipeline module.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import yaml


class TestVariantCreator:
    """Create different variants of test set for drift testing"""
    
    @staticmethod
    def modify_brightness(image: np.ndarray, factor: float = None) -> np.ndarray:
        """Modify image brightness"""
        if factor is None:
            factor = np.random.uniform(0.4, 1.8)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def modify_contrast(image: np.ndarray, alpha: float = None) -> np.ndarray:
        """Modify image contrast"""
        if alpha is None:
            alpha = np.random.uniform(0.5, 2.0)
        
        return np.clip(alpha * image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, std: float = 25) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def apply_motion_blur(image: np.ndarray, size: int = 15, angle: float = None) -> np.ndarray:
        """Apply motion blur"""
        if angle is None:
            angle = np.random.uniform(0, 180)
        
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = np.diag(np.ones(size))
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def simulate_rain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Simulate rain effect"""
        rain_drops = np.random.rand(*image.shape[:2])
        rain_mask = (rain_drops < intensity).astype(np.uint8) * 255
        rain_mask = cv2.GaussianBlur(rain_mask, (5, 5), 0)
        
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.add(result[:, :, i], rain_mask // 3)
        
        return result
    
    @staticmethod
    def simulate_fog(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Simulate fog effect"""
        fog = np.ones_like(image) * 255 * intensity
        return cv2.addWeighted(image, 1-intensity, fog.astype(np.uint8), intensity, 0)
    
    @staticmethod
    def apply_jpeg_compression(image: np.ndarray, quality: int = 30) -> np.ndarray:
        """Apply JPEG compression artifacts"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)
    
    def process_image(self, input_path: str, output_path: str, variant_type: str = None):
        """
        Process a single image with specified transformation
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            variant_type: Type of transformation (brightness, abstract, normal, etc.)
        """
        img = cv2.imread(input_path)
        if img is None:
            return
        
        if variant_type == 'brightness':
            img = self.modify_brightness(img)
        elif variant_type == 'contrast':
            img = self.modify_contrast(img)
        elif variant_type == 'blur':
            img = self.apply_gaussian_blur(img, kernel_size=15)
        elif variant_type == 'motion_blur':
            img = self.apply_motion_blur(img, size=15)
        elif variant_type == 'noise':
            img = self.add_gaussian_noise(img, std=25)
        elif variant_type == 'abstract':
            # Combination: blur + motion blur + noise
            img = self.apply_gaussian_blur(img, kernel_size=11)
            if np.random.rand() > 0.5:
                img = self.apply_motion_blur(img, size=11)
            img = self.add_gaussian_noise(img, std=15)
        elif variant_type == 'rain':
            img = self.simulate_rain(img, intensity=0.3)
        elif variant_type == 'fog':
            img = self.simulate_fog(img, intensity=0.5)
        elif variant_type == 'compression':
            img = self.apply_jpeg_compression(img, quality=30)
        # If variant_type is None or 'normal', keep original
        
        cv2.imwrite(output_path, img)
    
    def create_variants(self, input_dir: Path, output_dir: Path, 
                       variants: list = None) -> bool:
        """
        Create multiple test variants for drift testing
        
        Args:
            input_dir: Path to input test directory (should contain images/ and labels/)
            output_dir: Path to output base directory
            variants: List of variant types to create (default: ['normal', 'brightness', 'abstract'])
            
        Returns:
            bool: True if successful, False otherwise
        """
        if variants is None:
            variants = ['normal', 'brightness', 'abstract']
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            🔄 Test Variant Creator                               ║
╚══════════════════════════════════════════════════════════════════╝

Input:  {input_dir}
Output: {output_dir}
Variants: {', '.join(variants)}
""")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        descriptions = {
            'normal': 'Original test set without modifications',
            'brightness': 'Random brightness modifications',
            'brightness_dark': 'Darkened images (40% brightness)',
            'brightness_bright': 'Brightened images (160% brightness)',
            'contrast': 'Random contrast modifications',
            'blur': 'Gaussian blur applied',
            'motion_blur': 'Motion blur applied',
            'noise': 'Gaussian noise added',
            'abstract': 'Blur + motion blur + noise combined',
            'rain': 'Simulated rain effect',
            'fog': 'Simulated fog effect',
            'compression': 'JPEG compression artifacts',
            'severe': 'Multiple severe degradations combined'
        }
        
        # Create each variant
        success_count = 0
        for variant in variants:
            variant_output = output_dir / f'test_{variant}'
            success = create_variant(
                input_dir,
                variant_output,
                variant,
                descriptions.get(variant, '')
            )
            if success:
                success_count += 1
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ✅ Test Variants Created Successfully                           ║
╚══════════════════════════════════════════════════════════════════╝

📁 Output directory: {output_dir}

Created variants:
""")
        
        for variant in variants:
            variant_output = output_dir / f'test_{variant}'
            if (variant_output / 'images').exists():
                num_images = len(list((variant_output / 'images').glob('*')))
                print(f"  • test_{variant}/  ({num_images} images)")
        
        print(f"""
🔬 Use these variants for drift detection:
  python -m data_validation.drift_analysis \\
    --reference_dir {input_dir}/images \\
    --current_dir {output_dir}/test_brightness/images
""")
        
        return success_count == len(variants)


def create_variant(
    input_dir: Path,
    output_dir: Path,
    variant_type: str,
    description: str = ""
):
    """
    Create a test variant with specified transformations
    """
    print(f"\n🔄 Creating variant: {variant_type}")
    
    creator = TestVariantCreator()
    
    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'
    
    if not images_dir.exists():
        print(f"✗ Images directory not found: {images_dir}")
        return False
    
    # Create output directories
    output_images = output_dir / 'images'
    output_labels = output_dir / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy labels (unchanged)
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            shutil.copy(label_file, output_labels / label_file.name)
    
    # Process images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"  Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc=f"  {variant_type}"):
        img = cv2.imread(str(img_file))
        
        if img is None:
            continue
        
        # Apply transformation based on variant type
        if variant_type == 'normal':
            # No transformation
            pass
        
        elif variant_type == 'brightness':
            img = creator.modify_brightness(img)
        
        elif variant_type == 'brightness_dark':
            img = creator.modify_brightness(img, factor=0.4)
        
        elif variant_type == 'brightness_bright':
            img = creator.modify_brightness(img, factor=1.6)
        
        elif variant_type == 'contrast':
            img = creator.modify_contrast(img)
        
        elif variant_type == 'blur':
            img = creator.apply_gaussian_blur(img, kernel_size=15)
        
        elif variant_type == 'motion_blur':
            img = creator.apply_motion_blur(img, size=15)
        
        elif variant_type == 'noise':
            img = creator.add_gaussian_noise(img, std=25)
        
        elif variant_type == 'abstract':
            # Combination: blur + motion blur + noise
            img = creator.apply_gaussian_blur(img, kernel_size=11)
            if np.random.rand() > 0.5:
                img = creator.apply_motion_blur(img, size=11)
            img = creator.add_gaussian_noise(img, std=15)
        
        elif variant_type == 'rain':
            img = creator.simulate_rain(img, intensity=0.3)
        
        elif variant_type == 'fog':
            img = creator.simulate_fog(img, intensity=0.5)
        
        elif variant_type == 'compression':
            img = creator.apply_jpeg_compression(img, quality=30)
        
        elif variant_type == 'severe':
            # Multiple degradations
            img = creator.modify_brightness(img, factor=0.5)
            img = creator.apply_gaussian_blur(img, kernel_size=11)
            img = creator.add_gaussian_noise(img, std=30)
            img = creator.apply_jpeg_compression(img, quality=40)
        
        else:
            print(f"  ⚠️  Unknown variant type: {variant_type}")
        
        # Save processed image
        output_path = output_images / img_file.name
        cv2.imwrite(str(output_path), img)
    
    # Create metadata
    metadata = {
        'variant': variant_type,
        'description': description or f'{variant_type} transformation applied',
        'num_images': len(image_files),
        'source': str(input_dir.absolute())
    }
    
    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"  ✓ Created {variant_type}: {len(image_files)} images")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create test variants for drift testing')
    parser.add_argument('--input', required=True, help='Input test directory')
    parser.add_argument('--output', required=True, help='Output base directory')
    parser.add_argument('--variants', nargs='+', 
                       default=['normal', 'brightness', 'abstract'],
                       choices=['normal', 'brightness', 'brightness_dark', 'brightness_bright',
                               'contrast', 'blur', 'motion_blur', 'noise', 'abstract',
                               'rain', 'fog', 'compression', 'severe'],
                       help='Variants to create')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_base = Path(args.output)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            🔄 Test Variant Creator                               ║
╚══════════════════════════════════════════════════════════════════╝

Input:  {input_dir}
Output: {output_base}
Variants: {', '.join(args.variants)}
""")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create each variant
    for variant in args.variants:
        output_dir = output_base / f'test_{variant}'
        
        descriptions = {
            'normal': 'Original test set without modifications',
            'brightness': 'Random brightness modifications',
            'brightness_dark': 'Darkened images (40% brightness)',
            'brightness_bright': 'Brightened images (160% brightness)',
            'contrast': 'Random contrast modifications',
            'blur': 'Gaussian blur applied',
            'motion_blur': 'Motion blur applied',
            'noise': 'Gaussian noise added',
            'abstract': 'Blur + motion blur + noise combined',
            'rain': 'Simulated rain effect',
            'fog': 'Simulated fog effect',
            'compression': 'JPEG compression artifacts',
            'severe': 'Multiple severe degradations combined'
        }
        
        create_variant(
            input_dir,
            output_dir,
            variant,
            descriptions.get(variant, '')
        )
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ✅ Test Variants Created Successfully                           ║
╚══════════════════════════════════════════════════════════════════╝

📁 Output directory: {output_base}

Created variants:
""")
    
    for variant in args.variants:
        output_dir = output_base / f'test_{variant}'
        num_images = len(list((output_dir / 'images').glob('*')))
        print(f"  • test_{variant}/  ({num_images} images)")
    
    print(f"""
🔬 Use these variants for drift detection:
  python data_validation/drift_analysis.py \\
    --reference_dir {input_dir}/images \\
    --current_dir {output_base}/test_brightness/images
""")


if __name__ == "__main__":
    main()
