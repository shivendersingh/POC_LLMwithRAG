"""
Optimized RAG System with Performance Improvements

This module provides an optimized version of the RAG system with:
- Document caching to prevent duplicate processing
- File hash-based duplicate detection
- Optimized image processing
- Improved memory management
"""

import hashlib
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime
import asyncio

from rag_core import MockLLMClient, RAGSystem
from unified_pdf_processor import UnifiedPDFProcessor
from pdf_text_processor import SimplePDFTextProcessor, PDFQueryEngine, create_pdf_processor


class OptimizedRAGSystem:
    """
    Optimized RAG system with caching and duplicate prevention
    """
    
    def __init__(self, 
                 deepseek_api_key: str = None,
                 cache_dir: str = "./rag_cache",
                 max_context_chunks: int = 5):
        self.deepseek_api_key = deepseek_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_context_chunks = max_context_chunks
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache structures
        self.document_cache = {}  # file_hash -> processing results
        self.processed_files: Set[str] = set()  # Set of processed file hashes
        self.load_cache()
        
        # Initialize components
        self.llm_client = MockLLMClient()
        self.pdf_processor = None
        self.query_engine = None
        self.unified_processor = None
        
        # Performance tracking
        self.processing_stats = {
            "documents_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "duplicate_prevents": 0
        }
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating file hash: {str(e)}")
            # Fallback to filename + size + mtime
            stat = os.stat(file_path)
            fallback = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(fallback.encode()).hexdigest()
    
    def load_cache(self):
        """Load existing cache from disk"""
        cache_file = self.cache_dir / "document_cache.json"
        processed_file = self.cache_dir / "processed_files.json"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.document_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.document_cache)} cached documents")
            
            if processed_file.exists():
                with open(processed_file, 'r') as f:
                    self.processed_files = set(json.load(f))
                self.logger.info(f"Loaded {len(self.processed_files)} processed file hashes")
                
        except Exception as e:
            self.logger.warning(f"Could not load cache: {str(e)}")
            self.document_cache = {}
            self.processed_files = set()
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            cache_file = self.cache_dir / "document_cache.json"
            processed_file = self.cache_dir / "processed_files.json"
            
            with open(cache_file, 'w') as f:
                json.dump(self.document_cache, f, indent=2)
            
            with open(processed_file, 'w') as f:
                json.dump(list(self.processed_files), f, indent=2)
                
            self.logger.info("Cache saved successfully")
        except Exception as e:
            self.logger.error(f"Could not save cache: {str(e)}")
    
    def is_document_processed(self, file_path: str) -> bool:
        """Check if document has already been processed"""
        file_hash = self._get_file_hash(file_path)
        return file_hash in self.processed_files
    
    def get_cached_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached document processing results"""
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.document_cache:
            self.processing_stats["cache_hits"] += 1
            self.logger.info(f"Cache hit for document: {os.path.basename(file_path)}")
            return self.document_cache[file_hash]
        
        self.processing_stats["cache_misses"] += 1
        return None
    
    def cache_document(self, file_path: str, processing_results: Dict[str, Any]):
        """Cache document processing results"""
        file_hash = self._get_file_hash(file_path)
        
        # Store lightweight cache data
        cache_data = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "chunks_processed": processing_results.get("chunks_processed", 0),
            "images_processed": processing_results.get("images_processed", 0),
            "processing_time": processing_results.get("processing_time", 0),
            "status": processing_results.get("status", "unknown")
        }
        
        self.document_cache[file_hash] = cache_data
        self.processed_files.add(file_hash)
        
        # Save cache immediately for persistence
        self.save_cache()
        
        self.logger.info(f"Cached document: {os.path.basename(file_path)} ({file_hash[:8]}...)")
    
    async def initialize_processors(self):
        """Initialize PDF processors on demand"""
        if self.pdf_processor is None:
            self.logger.info("Initializing PDF processors...")
            
            # Create PDF processor (returns tuple, not coroutine)
            self.pdf_processor, self.query_engine = create_pdf_processor()
            
            # Create unified processor with proper components
            from unified_pdf_processor import UnifiedPDFProcessor
            
            # Create DeepSeek vision function
            async def vision_function(prompt: str, system_prompt: str, image_data: str) -> str:
                return await self.llm_client.describe_image(prompt, system_prompt, image_data)
            
            # Use the same embedding function and vector store as the text processor
            self.unified_processor = UnifiedPDFProcessor(
                llm_client=self.llm_client,
                vision_model_func=vision_function,
                embedding_func=self.pdf_processor.embedding_func,
                vector_store=self.pdf_processor.vector_store,
                text_processor=self.pdf_processor,
                image_processor=None,  # Will be created automatically
                enable_vqa=True  # Explicitly enable VQA
            )
            
            self.logger.info("PDF processors initialized successfully")
    
    async def process_document_optimized(self, 
                                       file_path: str, 
                                       chunk_size: int = 1000,
                                       chunk_overlap: int = 200,
                                       force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process document with optimization and caching
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            force_reprocess: Force reprocessing even if cached
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing document: {os.path.basename(file_path)}")
        
        # Check if already processed (unless forced)
        if not force_reprocess:
            if self.is_document_processed(file_path):
                self.processing_stats["duplicate_prevents"] += 1
                cached_result = self.get_cached_document(file_path)
                if cached_result:
                    self.logger.info(f"Document already processed, using cached result: {os.path.basename(file_path)}")
                    return {
                        "status": "cached",
                        "message": "Document was already processed",
                        "cached_data": cached_result,
                        "processing_time": 0
                    }
        
        # Initialize processors if needed
        await self.initialize_processors()
        
        try:
            start_time = datetime.now()
            
            # Process with unified processor for multimodal content
            self.logger.info("Starting unified multimodal processing...")
            processing_result = await self.unified_processor.process_pdf_complete(
                file_path, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert ProcessingResult dataclass to dictionary
            processing_results = {
                "status": processing_result.status,
                "doc_id": processing_result.doc_id,
                "chunks_processed": processing_result.text_chunks,
                "images_processed": processing_result.image_count,
                "processing_time": processing_time,
                "optimized": True,
                "cached": False,
                "errors": processing_result.errors or []
            }
            
            # Cache the results
            self.cache_document(file_path, processing_results)
            
            # Update stats
            self.processing_stats["documents_processed"] += 1
            
            self.logger.info(f"Document processing completed in {processing_time:.2f}s: {os.path.basename(file_path)}")
            
            return processing_results
            
        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def query_documents(self, 
                            question: str, 
                            max_results: int = 5,
                            similarity_threshold: float = 0.7,
                            query_mode: str = "auto",
                            enable_vqa: bool = True) -> Dict[str, Any]:
        """
        Query processed documents with VQA support and optimization
        
        Args:
            question: User question
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            query_mode: Query processing mode ("auto", "hybrid", "text_only", "image_only")
            enable_vqa: Whether to enable Visual Question Answering
            
        Returns:
            Query results and generated response with VQA support
        """
        self.logger.info(f"Processing query: {question[:50]}...")
        
        # Ensure processors are initialized
        await self.initialize_processors()
        
        if self.unified_processor is None:
            return {
                "status": "error",
                "error": "No documents have been processed yet. Please add documents first."
            }
        
        try:
            start_time = datetime.now()
            
            # Adjust query mode if VQA is disabled
            effective_mode = query_mode
            if not enable_vqa and query_mode in ["auto", "image_only", "hybrid"]:
                # If VQA is disabled, force text-only mode for auto/image/hybrid modes
                effective_mode = "text_only" if query_mode in ["auto", "hybrid"] else "text_only"
                self.logger.info(f"VQA disabled, using mode: {effective_mode}")
            
            # Use the unified processor with VQA support
            result = await self.unified_processor.query(
                query_text=question,
                top_k=max_results,
                mode=effective_mode,
                max_tokens=1000,
                temperature=0.7
            )
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            # Enhance the result with additional metadata and map fields for Streamlit compatibility
            if result.get("status") == "success":
                # Map unified processor response to expected Streamlit format
                result.update({
                    "answer": result.get("response", ""),  # Map 'response' to 'answer'
                    "query_time": query_time,
                    "context_chunks": result.get("context_used", 0),  # Map 'context_used' to 'context_chunks'
                    "search_results": result.get("sources", []),  # Map 'sources' to 'search_results'
                    "used_vqa": result.get("query_mode") == "visual_qa",  # Check if VQA was used
                    "query_mode": result.get("query_mode", effective_mode),
                    "document_type": "multimodal",  # Default for unified processor
                    "text_sources": result.get("text_sources", 0),
                    "image_sources": result.get("image_sources", 0)
                })
                self.logger.info(f"Query completed in {query_time:.2f}s")
            else:
                result["query_time"] = query_time
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            query_time = (datetime.now() - start_time).total_seconds()
            return {
                "status": "error",
                "error": error_msg,
                "query_time": query_time
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_operations = self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]
        cache_hit_rate = (self.processing_stats["cache_hits"] / total_operations * 100) if total_operations > 0 else 0
        
        return {
            **self.processing_stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_cached_documents": len(self.document_cache),
            "total_processed_files": len(self.processed_files)
        }
    
    def clear_cache(self, confirm: bool = False):
        """Clear all cached data"""
        if not confirm:
            self.logger.warning("Cache clear requested but not confirmed")
            return False
        
        self.document_cache.clear()
        self.processed_files.clear()
        
        # Remove cache files
        try:
            cache_file = self.cache_dir / "document_cache.json"
            processed_file = self.cache_dir / "processed_files.json"
            
            if cache_file.exists():
                cache_file.unlink()
            if processed_file.exists():
                processed_file.unlink()
                
            self.logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False


# Convenience function to create optimized RAG system
def create_optimized_rag_system(deepseek_api_key: str = None, 
                               cache_dir: str = "./rag_cache") -> OptimizedRAGSystem:
    """Create an optimized RAG system instance"""
    return OptimizedRAGSystem(
        deepseek_api_key=deepseek_api_key,
        cache_dir=cache_dir
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize optimized RAG system
        rag_system = create_optimized_rag_system("your-deepseek-api-key")
        
        # Process a document
        result = await rag_system.process_document_optimized("example.pdf")
        print(f"Processing result: {result}")
        
        # Query the documents
        query_result = await rag_system.query_documents("What is this document about?")
        print(f"Query result: {query_result}")
        
        # Get performance stats
        stats = rag_system.get_processing_stats()
        print(f"Performance stats: {stats}")
    
    asyncio.run(main())