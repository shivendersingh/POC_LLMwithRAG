"""
Image Repository for Visual Question Answering (VQA)

This module provides storage and management of images for direct VQA access,
enabling the system to re-analyze images with specific questions.
"""

import os
import base64
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ImageRepository:
    """
    Repository for storing and managing images for Visual Question Answering
    
    Stores image data and metadata to enable direct questioning of specific images
    without relying solely on pre-generated descriptions.
    """
    
    def __init__(self, base_dir: str = "./image_storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "image_metadata.json"
        self.image_metadata = {}  # doc_id -> list of image metadata
        self.logger = logging.getLogger(__name__)
        
        # Load existing metadata
        self.load_metadata()
    
    def load_metadata(self):
        """Load existing image metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.image_metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self.image_metadata)} documents")
        except Exception as e:
            self.logger.warning(f"Could not load image metadata: {str(e)}")
            self.image_metadata = {}
    
    def save_metadata(self):
        """Save image metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.image_metadata, f, indent=2)
            self.logger.debug("Image metadata saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save image metadata: {str(e)}")
    
    def store_image(self, 
                   image_data: str, 
                   doc_id: str, 
                   image_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store image data for later VQA use
        
        Args:
            image_data: Base64 encoded image data
            doc_id: Document identifier
            image_id: Unique image identifier
            metadata: Additional metadata for the image
            
        Returns:
            Path to stored image file
        """
        try:
            # Create document directory
            doc_dir = self.base_dir / doc_id
            doc_dir.mkdir(exist_ok=True)
            
            # Save image file
            image_path = doc_dir / f"{image_id}.png"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            
            # Store metadata
            if doc_id not in self.image_metadata:
                self.image_metadata[doc_id] = []
            
            image_metadata = {
                "image_id": image_id,
                "path": str(image_path),
                "stored_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "embedding_id": metadata.get("embedding_id") if metadata else None
            }
            
            self.image_metadata[doc_id].append(image_metadata)
            
            # Save metadata to disk
            self.save_metadata()
            
            self.logger.info(f"Stored image: {image_id} for document {doc_id}")
            return str(image_path)
            
        except Exception as e:
            self.logger.error(f"Failed to store image {image_id}: {str(e)}")
            raise
    
    def get_image_data(self, doc_id: str, image_id: str) -> Optional[str]:
        """
        Get base64 image data for VQA
        
        Args:
            doc_id: Document identifier
            image_id: Image identifier
            
        Returns:
            Base64 encoded image data or None if not found
        """
        try:
            image_path = self.base_dir / doc_id / f"{image_id}.png"
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return None
                
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Failed to get image data {doc_id}/{image_id}: {str(e)}")
            return None
    
    def get_images_by_embeddings(self, embedding_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get images associated with retrieved embeddings
        
        Args:
            embedding_ids: List of embedding IDs to look up
            
        Returns:
            List of image data and metadata dictionaries
        """
        images = []
        try:
            for doc_id, doc_images in self.image_metadata.items():
                for img in doc_images:
                    if img.get("embedding_id") in embedding_ids:
                        img_data = self.get_image_data(doc_id, img["image_id"])
                        if img_data:
                            images.append({
                                "image_data": img_data,
                                "metadata": img["metadata"],
                                "doc_id": doc_id,
                                "image_id": img["image_id"]
                            })
            
            self.logger.info(f"Retrieved {len(images)} images for {len(embedding_ids)} embeddings")
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to get images by embeddings: {str(e)}")
            return []
    
    def get_all_images_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all images for a specific document
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of image data and metadata dictionaries
        """
        images = []
        try:
            if doc_id not in self.image_metadata:
                return images
            
            for img in self.image_metadata[doc_id]:
                img_data = self.get_image_data(doc_id, img["image_id"])
                if img_data:
                    images.append({
                        "image_data": img_data,
                        "metadata": img["metadata"],
                        "doc_id": doc_id,
                        "image_id": img["image_id"]
                    })
            
            self.logger.info(f"Retrieved {len(images)} images for document {doc_id}")
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to get images for document {doc_id}: {str(e)}")
            return []
    
    def list_documents(self) -> List[str]:
        """Get list of document IDs with stored images"""
        return list(self.image_metadata.keys())
    
    def get_document_image_count(self, doc_id: str) -> int:
        """Get number of images stored for a document"""
        return len(self.image_metadata.get(doc_id, []))
    
    def get_total_image_count(self) -> int:
        """Get total number of stored images across all documents"""
        return sum(len(images) for images in self.image_metadata.values())
    
    def cleanup_document(self, doc_id: str) -> bool:
        """
        Remove all images and metadata for a document
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id not in self.image_metadata:
                self.logger.info(f"No images found for document {doc_id}")
                return True
            
            # Remove files
            doc_dir = self.base_dir / doc_id
            if doc_dir.exists():
                import shutil
                shutil.rmtree(doc_dir)
            
            # Remove metadata
            del self.image_metadata[doc_id]
            self.save_metadata()
            
            self.logger.info(f"Cleaned up images for document {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup document {doc_id}: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_images = self.get_total_image_count()
        total_docs = len(self.image_metadata)
        
        # Calculate storage size
        total_size = 0
        try:
            for doc_dir in self.base_dir.iterdir():
                if doc_dir.is_dir():
                    for img_file in doc_dir.glob("*.png"):
                        total_size += img_file.stat().st_size
        except Exception as e:
            self.logger.warning(f"Could not calculate storage size: {str(e)}")
            total_size = -1
        
        return {
            "total_images": total_images,
            "total_documents": total_docs,
            "storage_size_bytes": total_size,
            "storage_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else -1,
            "base_directory": str(self.base_dir)
        }


# Test function
async def test_image_repository():
    """Test the ImageRepository functionality"""
    repo = ImageRepository("./test_image_storage")
    
    # Create a test image (simple base64 encoded image)
    test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Store test image
    doc_id = "test_doc_123"
    image_id = "test_img_001"
    metadata = {
        "description": "Test image for VQA",
        "embedding_id": "emb_123",
        "source": "test.pdf"
    }
    
    try:
        path = repo.store_image(test_image_data, doc_id, image_id, metadata)
        print(f"Image stored at: {path}")
        
        # Retrieve image
        retrieved_data = repo.get_image_data(doc_id, image_id)
        print(f"Retrieved image data length: {len(retrieved_data) if retrieved_data else 0}")
        
        # Get storage stats
        stats = repo.get_storage_stats()
        print(f"Storage stats: {stats}")
        
        # Cleanup
        success = repo.cleanup_document(doc_id)
        print(f"Cleanup successful: {success}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_image_repository())