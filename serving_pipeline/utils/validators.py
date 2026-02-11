"""
Image validation utilities.
"""
import io
import imghdr
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from fastapi import UploadFile, HTTPException, status

from ..config import settings


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ImageValidator:
    """Validator for image files."""
    
    def __init__(
        self,
        allowed_extensions: list = None,
        max_file_size_mb: float = None,
        min_width: int = None,
        min_height: int = None,
        max_width: int = None,
        max_height: int = None
    ):
        """
        Initialize the image validator.
        
        Args:
            allowed_extensions: List of allowed file extensions (e.g., ['.jpg', '.png'])
            max_file_size_mb: Maximum file size in megabytes
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            max_width: Maximum image width in pixels
            max_height: Maximum image height in pixels
        """
        self.allowed_extensions = allowed_extensions or settings.ALLOWED_EXTENSIONS
        self.max_file_size_mb = max_file_size_mb or settings.MAX_FILE_SIZE_MB
        self.min_width = min_width or settings.MIN_IMAGE_WIDTH
        self.min_height = min_height or settings.MIN_IMAGE_HEIGHT
        self.max_width = max_width or settings.MAX_IMAGE_WIDTH
        self.max_height = max_height or settings.MAX_IMAGE_HEIGHT
        self.max_file_size_bytes = int(self.max_file_size_mb * 1024 * 1024)
    
    def validate_extension(self, filename: str) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if extension is valid
            
        Raises:
            ValidationError: If extension is not allowed
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise ValidationError(
                f"File extension '{ext}' is not allowed. "
                f"Allowed extensions: {', '.join(self.allowed_extensions)}"
            )
        return True
    
    def validate_file_size(self, file_size: int) -> bool:
        """
        Validate file size.
        
        Args:
            file_size: Size of the file in bytes
            
        Returns:
            True if size is valid
            
        Raises:
            ValidationError: If file is too large
        """
        if file_size > self.max_file_size_bytes:
            raise ValidationError(
                f"File size ({file_size / (1024*1024):.2f}MB) exceeds "
                f"maximum allowed size ({self.max_file_size_mb}MB)"
            )
        return True
    
    def validate_image_content(self, image_bytes: bytes) -> Tuple[int, int, str]:
        """
        Validate image content and get dimensions.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (width, height, format)
            
        Raises:
            ValidationError: If image is invalid or dimensions are out of range
        """
        # Check if it's a valid image
        image_type = imghdr.what(None, h=image_bytes)
        if image_type is None:
            raise ValidationError("File is not a valid image")
        
        # Open image and get dimensions
        try:
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
        except Exception as e:
            raise ValidationError(f"Failed to read image: {str(e)}")
        
        # Validate dimensions
        if width < self.min_width or height < self.min_height:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too small. "
                f"Minimum size: {self.min_width}x{self.min_height}"
            )
        
        if width > self.max_width or height > self.max_height:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too large. "
                f"Maximum size: {self.max_width}x{self.max_height}"
            )
        
        return width, height, image_type
    
    async def validate_upload_file(self, file: UploadFile) -> Tuple[bytes, int, int, str]:
        """
        Validate an uploaded file.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (image_bytes, width, height, format)
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Validate extension
            self.validate_extension(file.filename)
            
            # Read file content
            content = await file.read()
            
            # Validate file size
            self.validate_file_size(len(content))
            
            # Validate image content
            width, height, image_format = self.validate_image_content(content)
            
            # Reset file position
            await file.seek(0)
            
            return content, width, height, image_format
            
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.message
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error validating image: {str(e)}"
            )
    
    def validate_pil_image(self, image: Image.Image) -> Tuple[int, int]:
        """
        Validate a PIL Image object.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            ValidationError: If validation fails
        """
        width, height = image.size
        
        if width < self.min_width or height < self.min_height:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too small. "
                f"Minimum size: {self.min_width}x{self.min_height}"
            )
        
        if width > self.max_width or height > self.max_height:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are too large. "
                f"Maximum size: {self.max_width}x{self.max_height}"
            )
        
        return width, height
