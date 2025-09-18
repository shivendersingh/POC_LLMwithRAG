"""
PDF Image Processing Module for RAG System

This module implements SimplePDFImageProcessor for extracting images from PDF documents,
generating descriptions using vision models, and storing them in vector databases for 
multimodal RAG applications.
"""

import logging
import uuid
import asyncio
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod


class VisionModelFunction(ABC):
    """Abstract base class for vision model functions"""
    
    @abstractmethod
    async def __call__(self, prompt: str, system_prompt: str, image_data: str) -> str:
        """Generate description for an image"""
        pass


class SimplePDFImageProcessor:
    """
    Core PDF image processing class for RAG applications.
    
    Handles PDF image extraction, vision model description generation,
    embedding creation, and vector storage for retrieval-augmented generation.
    """
    
    def __init__(self, vision_model_func, embedding_func, vector_store, image_repository=None):
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        self.image_repository = image_repository
        self.logger = logging.getLogger(__name__)
    
    async def process_pdf_images(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract and process images from PDF
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Starting PDF image processing for: {pdf_path}")
            
            # 1. Extract images from PDF
            image_paths = await self._extract_images_from_pdf(pdf_path, output_dir)
            self.logger.info(f"Extracted {len(image_paths)} images from PDF")
            
            if not image_paths:
                return {
                    "status": "success",
                    "images_processed": 0,
                    "message": "No images found in PDF",
                    "document_path": pdf_path
                }
            
            # 2. Process each image
            processed_count = 0
            descriptions = []
            
            for img_path in image_paths:
                try:
                    description = await self._process_single_image(img_path, source_file=pdf_path)
                    descriptions.append(description)
                    processed_count += 1
                    self.logger.info(f"Processed image {processed_count}/{len(image_paths)}: {os.path.basename(img_path)}")
                except Exception as e:
                    self.logger.error(f"Failed to process image {img_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully processed {processed_count}/{len(image_paths)} images")
            
            return {
                "status": "success",
                "images_processed": processed_count,
                "total_images": len(image_paths),
                "descriptions": descriptions,
                "document_path": pdf_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF images: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "images_processed": 0,
                "document_path": pdf_path
            }
    
    async def _extract_images_from_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract images from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to extracted images
        """
        self.logger.info(f"Extracting images from PDF: {pdf_path}")
        
        # Create output directory if needed
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        
        try:
            # Open PDF document
            pdf_document = fitz.open(pdf_path)
            self.logger.info(f"PDF has {len(pdf_document)} pages")
            
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                image_list = page.get_images(full=True)
                
                self.logger.info(f"Page {page_index + 1} contains {len(image_list)} images")
                
                for image_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Skip very small images (likely artifacts)
                        if len(image_bytes) < 1024:  # Less than 1KB
                            self.logger.debug(f"Skipping small image: {len(image_bytes)} bytes")
                            continue
                        
                        # Create image filename
                        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                        image_filename = f"{pdf_name}_page{page_index+1}_img{image_index+1}.{base_image['ext']}"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        # Save image
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_bytes)
                        
                        # Verify image is valid and meets quality requirements
                        if self._validate_and_filter_image(image_path):
                            image_paths.append(image_path)
                            self.logger.info(f"Extracted image: {image_filename}")
                        else:
                            if os.path.exists(image_path):
                                os.remove(image_path)  # Remove invalid/filtered image
                            
                    except Exception as e:
                        self.logger.error(f"Failed to extract image {image_index} from page {page_index}: {str(e)}")
                        continue
            
            pdf_document.close()
            
        except Exception as e:
            self.logger.error(f"Failed to extract images from PDF: {str(e)}")
            raise
        
        return image_paths
    
    def _validate_and_filter_image(self, image_path: str) -> bool:
        """
        Validate and filter images based on quality and relevance criteria
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image passes all filters, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Basic validation - verify image integrity
                img.verify()
                
                # Re-open for dimension analysis (verify() closes the image)
                with Image.open(image_path) as img_for_analysis:
                    width, height = img_for_analysis.size
                    
                    # Filter 1: Skip very small images (likely logos, icons, artifacts)
                    min_dimension = 50  # pixels
                    if width < min_dimension or height < min_dimension:
                        self.logger.debug(f"Skipping small image: {width}x{height} pixels")
                        return False
                    
                    # Filter 2: Skip very thin/wide images (likely decorative elements)
                    aspect_ratio = max(width, height) / min(width, height)
                    max_aspect_ratio = 10.0  # Skip images more than 10:1 ratio
                    if aspect_ratio > max_aspect_ratio:
                        self.logger.debug(f"Skipping image with extreme aspect ratio: {aspect_ratio:.2f}")
                        return False
                    
                    # Filter 3: Skip very small area images
                    min_area = 2500  # 50x50 pixels minimum area
                    if width * height < min_area:
                        self.logger.debug(f"Skipping image with small area: {width * height} pixels")
                        return False
                    
                    # Filter 4: Check file size on disk
                    file_size = os.path.getsize(image_path)
                    min_file_size = 2048  # 2KB minimum
                    max_file_size = 10 * 1024 * 1024  # 10MB maximum
                    
                    if file_size < min_file_size:
                        self.logger.debug(f"Skipping image with small file size: {file_size} bytes")
                        return False
                    
                    if file_size > max_file_size:
                        self.logger.debug(f"Skipping very large image: {file_size} bytes")
                        return False
                    
                    # Image passes all filters
                    self.logger.debug(f"Image passed filters: {width}x{height}, {file_size} bytes, ratio: {aspect_ratio:.2f}")
                    return True
                    
        except Exception as e:
            self.logger.debug(f"Image validation failed: {str(e)}")
            return False
    
    async def _process_single_image(self, image_path: str, source_file: str) -> str:
        """
        Process a single image with vision model and OCR
        
        Args:
            image_path: Path to image file
            source_file: Source PDF file path
            
        Returns:
            Generated image description enhanced with OCR text
        """
        self.logger.info(f"Processing image: {os.path.basename(image_path)}")
        
        try:
            # 1. Convert image to base64
            image_data = self._encode_image_to_base64(image_path)
            
            # 2. Extract text from image using OCR
            ocr_text = await self._extract_text_with_ocr(image_path)
            
            # 3. Enhance prompt with OCR results
            if ocr_text.strip():
                enhanced_prompt = f"""Describe this image in detail, focusing on key visual elements, diagrams, charts, and any relevant information.
                
                The image contains the following text (extracted via OCR): "{ocr_text}"
                
                Please incorporate this text information into your description and explain how it relates to the visual elements.
                Focus on content that would be useful for document search and retrieval."""
                
                system_prompt = "You are an AI assistant that provides detailed, accurate descriptions of images from documents. You have access to OCR text from the image. Combine visual analysis with text content for comprehensive descriptions."
            else:
                enhanced_prompt = "Describe this image in detail, focusing on key visual elements, text content, diagrams, charts, and any relevant information that would be useful for document search and retrieval."
                system_prompt = "You are an AI assistant that provides detailed, accurate descriptions of images from documents. Focus on content that would be useful for search and question-answering."
            
            description = await self.vision_model_func(
                prompt=enhanced_prompt,
                system_prompt=system_prompt,
                image_data=image_data
            )
            
            # 5. Append OCR text to description if it wasn't incorporated
            if ocr_text.strip() and "OCR extracted text:" not in description and ocr_text not in description:
                description += f"\n\nText detected in image (OCR): {ocr_text}"
            
            embeddings = await self.embedding_func([description])
            
            image_id = f"img_{uuid.uuid4().hex[:16]}"
            
            # Generate unique embedding ID for VQA linking
            embedding_id = f"img_emb_{uuid.uuid4().hex}"
            
            self.vector_store.add_embedding(
                embedding=embeddings[0],
                text=description,
                metadata={
                    "id": image_id,
                    "source": source_file,
                    "image_path": image_path,
                    "content_type": "image",
                    "image_filename": os.path.basename(image_path),
                    "has_ocr_text": bool(ocr_text.strip()),
                    "ocr_text": ocr_text if ocr_text.strip() else None,
                    "embedding_id": embedding_id
                }
            )
            
            # Store image in repository for direct VQA access
            if self.image_repository:
                try:
                    doc_id = os.path.splitext(os.path.basename(source_file))[0]
                    repo_image_id = os.path.splitext(os.path.basename(image_path))[0]
                    
                    self.image_repository.store_image(
                        image_data=image_data,
                        doc_id=doc_id,
                        image_id=repo_image_id,
                        metadata={
                            "description": description,
                            "source": source_file,
                            "embedding_id": embedding_id,
                            "has_ocr_text": bool(ocr_text.strip()),
                            "ocr_text": ocr_text if ocr_text.strip() else None,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    self.logger.info(f"Stored image in repository for VQA: {repo_image_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to store image in repository: {str(e)}")
            
            self.logger.info(f"Successfully processed and stored image: {os.path.basename(image_path)}")
            
            return description
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {str(e)}")
            raise
    
    async def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR (pytesseract)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text string
        """
        try:
            # Try to import pytesseract
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                self.logger.warning("pytesseract not available, skipping OCR")
                return ""
            
            # Open and process image for OCR
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                
                # Clean up the text
                cleaned_text = ' '.join(ocr_text.split())  # Remove extra whitespace
                
                if cleaned_text.strip():
                    self.logger.info(f"OCR extracted text from {os.path.basename(image_path)}: {cleaned_text[:100]}...")
                    return cleaned_text
                else:
                    self.logger.debug(f"No text found in image: {os.path.basename(image_path)}")
                    return ""
                    
        except Exception as e:
            self.logger.warning(f"OCR failed for {image_path}: {str(e)}")
            return ""

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Convert image to base64 encoding
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image to base64: {str(e)}")
            raise
    
    async def query_images(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query for relevant images based on text query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant image results with metadata
        """
        self.logger.info(f"Querying images for: {query}")
        
        try:
            # Generate embedding for query
            query_embeddings = await self.embedding_func([query])
            query_embedding = query_embeddings[0]
            
            # Search vector store for similar image descriptions
            results = self.vector_store.search(query_embedding, k=k)
            
            # Filter for image content only
            image_results = [
                result for result in results 
                if result.get('metadata', {}).get('content_type') == 'image'
            ]
            
            self.logger.info(f"Found {len(image_results)} relevant images")
            
            return image_results
            
        except Exception as e:
            self.logger.error(f"Error querying images: {str(e)}")
            return []


class PDFImageQueryEngine:
    """
    Query engine for PDF image content
    """
    
    def __init__(self, image_processor: SimplePDFImageProcessor):
        self.image_processor = image_processor
        self.logger = logging.getLogger(__name__)
    
    async def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Query PDF images for relevant content
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Query results with image descriptions and metadata
        """
        try:
            results = await self.image_processor.query_images(query, k=k)
            
            if not results:
                return {
                    "status": "success",
                    "results": [],
                    "message": "No relevant images found"
                }
            
            return {
                "status": "success",
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in image query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }


def create_pdf_image_processor(vision_model_func, embedding_func, vector_store, image_repository=None) -> tuple[SimplePDFImageProcessor, PDFImageQueryEngine]:
    """
    Factory function to create PDF image processor and query engine
    
    Args:
        vision_model_func: Vision model function for image description
        embedding_func: Embedding function for text vectorization
        vector_store: Vector store for storing embeddings
        image_repository: Optional ImageRepository for VQA support
        
    Returns:
        Tuple of (SimplePDFImageProcessor, PDFImageQueryEngine)
    """
    processor = SimplePDFImageProcessor(vision_model_func, embedding_func, vector_store, image_repository)
    query_engine = PDFImageQueryEngine(processor)
    
    return processor, query_engine


# Test function for validation
async def test_image_extraction(pdf_path: str):
    """
    Test function to validate image extraction from PDF
    
    Args:
        pdf_path: Path to test PDF file
    """
    from sentence_transformers import SentenceTransformer
    from pdf_text_processor import ChromaVectorStore
    
    # Mock vision model function for testing
    async def mock_vision_model(prompt: str, system_prompt: str, image_data: str) -> str:
        return f"Mock description for image with {len(image_data)} base64 characters"
    
    # Mock embedding function
    model = SentenceTransformer('all-MiniLM-L6-v2')
    async def mock_embedding_func(texts: List[str]) -> List[List[float]]:
        return model.encode(texts).tolist()
    
    # Create vector store
    vector_store = ChromaVectorStore(collection_name="test_images")
    
    # Create processor
    processor, query_engine = create_pdf_image_processor(
        mock_vision_model, mock_embedding_func, vector_store
    )
    
    # Test image extraction
    result = await processor.process_pdf_images(pdf_path)
    print(f"Image processing result: {result}")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = asyncio.run(test_image_extraction(pdf_path))
        print(f"Processed {result.get('images_processed', 0)} images")
    else:
        print("Usage: python pdf_image_processor.py <pdf_path>")