"""
Image Extraction and Understanding Module

Extracts images from PDFs and provides OCR + vision-language understanding.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not available, image extraction disabled")

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logger.warning("pytesseract not available, OCR disabled")

try:
    from sentence_transformers import SentenceTransformer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("sentence-transformers not available, CLIP embeddings disabled")


def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract images from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of image dicts with:
            - image: PIL Image object
            - page_number: Page number (1-indexed)
            - bbox: Bounding box (x0, y0, x1, y1)
    """
    if not HAS_PIL:
        logger.warning("Pillow not available, cannot extract images")
        return []
    
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    images.append({
                        'image': image,
                        'page_number': page_num + 1,
                        'image_index': img_index,
                        'width': image.width,
                        'height': image.height,
                        'format': base_image.get("ext", "unknown"),
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        logger.info(f"Extracted {len(images)} images from PDF")
        return images
        
    except Exception as e:
        logger.error(f"Failed to extract images from PDF: {e}")
        return []


def ocr_image(image: Image.Image) -> str:
    """
    Extract text from image using OCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text
    """
    if not HAS_OCR:
        logger.warning("pytesseract not available, OCR disabled")
        return ""
    
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def get_image_embedding(
    image: Image.Image,
    model: Optional[SentenceTransformer] = None,
) -> Optional[List[float]]:
    """
    Get CLIP embedding for image (requires CLIP model).
    
    Args:
        image: PIL Image object
        model: Optional pre-loaded CLIP model
        
    Returns:
        Image embedding vector or None
    """
    if not HAS_CLIP:
        logger.warning("sentence-transformers not available, CLIP embeddings disabled")
        return None
    
    # Note: This requires a CLIP model, which is different from sentence-transformers
    # For now, return None - full CLIP implementation would require additional setup
    logger.warning("CLIP embeddings not fully implemented (requires CLIP model)")
    return None


def process_standalone_image(
    image_path: str,
    extract_ocr: bool = True,
    extract_embeddings: bool = False,
) -> Dict:
    """
    Process a standalone image file: OCR and optionally embed.
    
    Args:
        image_path: Path to image file
        extract_ocr: Whether to extract text using OCR
        extract_embeddings: Whether to generate image embeddings
        
    Returns:
        Processed image dict with OCR text and metadata
    """
    if not HAS_PIL:
        raise ImportError("Pillow is required for image processing")
    
    try:
        image = Image.open(image_path)
        
        # Extract OCR text
        ocr_text = ""
        if extract_ocr and HAS_OCR:
            ocr_text = ocr_image(image)
        
        # Get embedding if requested
        embedding = None
        if extract_embeddings:
            embedding = get_image_embedding(image)
        
        return {
            'width': image.width,
            'height': image.height,
            'format': image.format or 'unknown',
            'ocr_text': ocr_text,
            'has_text': len(ocr_text) > 0,
            'embedding': embedding,
            'image_path': image_path,
        }
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        raise


def process_pdf_images(
    pdf_path: str,
    extract_ocr: bool = True,
    extract_embeddings: bool = False,
) -> List[Dict]:
    """
    Process all images from a PDF: extract, OCR, and optionally embed.
    
    Args:
        pdf_path: Path to PDF file
        extract_ocr: Whether to extract text using OCR
        extract_embeddings: Whether to generate image embeddings
        
    Returns:
        List of processed image dicts with OCR text and metadata
    """
    images = extract_images_from_pdf(pdf_path)
    
    processed = []
    for img_data in images:
        image = img_data['image']
        
        # Extract OCR text
        ocr_text = ""
        if extract_ocr:
            ocr_text = ocr_image(image)
        
        # Get embedding if requested
        embedding = None
        if extract_embeddings:
            embedding = get_image_embedding(image)
        
        processed.append({
            'page_number': img_data['page_number'],
            'image_index': img_data['image_index'],
            'width': img_data['width'],
            'height': img_data['height'],
            'format': img_data['format'],
            'ocr_text': ocr_text,
            'has_text': len(ocr_text) > 0,
            'embedding': embedding,
        })
    
    logger.info(f"Processed {len(processed)} images from PDF")
    return processed


__all__ = [
    "extract_images_from_pdf",
    "ocr_image",
    "process_pdf_images",
    "process_standalone_image",
]

