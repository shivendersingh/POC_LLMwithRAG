"""
Task C: Unified PDF Processor for Multimodal RAG
PDF with Word and Image Processing (Multimodal)

This module implements a unified processor that            # Store document metadata for smart query mode selection
            self.document_metadata[doc_id] = {
                "doc_type": doc_type,
                "text_chunks": text_chunks,
                "image_count": image_count,
                "file_path": pdf_path,
                "processed_at": datetime.now().isoformat()
            }
            
            result = ProcessingResult(
                doc_id=doc_id,
                text_chunks=text_chunks,
                image_count=image_count,
                processing_time=processing_time,
                doc_type=doc_type,
                status=status,
                errors=errors
            )
            
            self.logger.info(f"Unified processing completed: {result}")
            return resultxt and image processing
from Tasks A and B for comprehensive multimodal document understanding.
"""

import logging
import uuid
import asyncio
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Import from existing implementations
from pdf_text_processor import SimplePDFTextProcessor, PDFQueryEngine, create_pdf_processor
from pdf_image_processor import SimplePDFImageProcessor, PDFImageQueryEngine, create_pdf_image_processor
from rag_core import MockLLMClient
from image_repository import ImageRepository
from visual_qa_engine import VisualQAEngine


@dataclass
class ProcessingResult:
    """Result of processing operations"""
    doc_id: str
    text_chunks: int
    image_count: int
    processing_time: float
    status: str
    doc_type: str = "unknown"  # image_only, text_only, multimodal, empty
    errors: List[str] = None


@dataclass
class SearchResult:
    """Search result from vector store"""
    text: str
    metadata: dict
    score: float
    content_type: str


class UnifiedPDFProcessor:
    """
    Unified PDF processor that handles both text and image content
    for comprehensive multimodal RAG applications.
    
    This class combines the functionality of Tasks A and B to provide
    a complete solution for processing PDFs with mixed content.
    """
    
    def __init__(self,
                 llm_client,  # DeepSeek text model client
                 vision_model_func,  # DeepSeek vision model function
                 embedding_func,  # Embedding function
                 vector_store,  # Unified vector store
                 text_processor: Optional[SimplePDFTextProcessor] = None,
                 image_processor: Optional[SimplePDFImageProcessor] = None,
                 image_repository: Optional[ImageRepository] = None,
                 enable_vqa: bool = True):
        
        self.llm_client = llm_client
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        
        # Initialize image repository for VQA
        self.image_repository = image_repository or ImageRepository() if enable_vqa else None
        
        # Use provided processors or create new ones
        self.text_processor = text_processor or SimplePDFTextProcessor(embedding_func, vector_store)
        self.image_processor = image_processor or SimplePDFImageProcessor(
            vision_model_func, embedding_func, vector_store, self.image_repository
        )
        
        # Initialize VQA engine if enabled
        self.vqa_engine = None
        if enable_vqa and self.image_repository:
            self.vqa_engine = VisualQAEngine(
                vision_model_func=vision_model_func,
                image_repository=self.image_repository,
                vector_store=vector_store,
                embedding_func=embedding_func
            )
        
        self.logger = logging.getLogger(__name__)
        
        # Store document metadata for smart query mode selection
        self.document_metadata = {}  # doc_id -> metadata
    
    async def process_pdf_complete(self, 
                                 pdf_path: str, 
                                 output_dir: Optional[str] = None,
                                 chunk_size: int = 1000, 
                                 chunk_overlap: int = 200,
                                 process_text: bool = True,
                                 process_images: bool = True) -> ProcessingResult:
        """
        Process PDF with both text and images in a unified pipeline
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory for extracted images
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            process_text: Whether to process text content
            process_images: Whether to process image content
            
        Returns:
            ProcessingResult with comprehensive processing information
        """
        start_time = datetime.now()
        doc_id = f"doc_{uuid.uuid4().hex}"
        errors = []
        
        self.logger.info(f"Starting unified PDF processing for: {pdf_path}")
        self.logger.info(f"Document ID: {doc_id}")
        
        try:
            text_chunks = 0
            image_count = 0
            
            # Process text content
            if process_text:
                try:
                    self.logger.info("Processing text content...")
                    text_chunks = await self.text_processor.process_pdf_text(
                        pdf_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    self.logger.info(f"Text processing completed: {text_chunks} chunks")
                except Exception as e:
                    error_msg = f"Text processing failed: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Process image content
            if process_images:
                try:
                    self.logger.info("Processing image content...")
                    image_result = await self.image_processor.process_pdf_images(
                        pdf_path,
                        output_dir=output_dir
                    )
                    
                    if image_result.get("status") == "success":
                        image_count = image_result.get("images_processed", 0)
                        self.logger.info(f"Image processing completed: {image_count} images")
                    else:
                        error_msg = f"Image processing failed: {image_result.get('error', 'Unknown error')}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Image processing failed: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Determine document type based on content
            doc_type = self._determine_document_type(text_chunks, image_count)
            self.logger.info(f"Document type detected: {doc_type}")
            
            # Determine overall status
            status = "success" if not errors else "partial_success" if (text_chunks > 0 or image_count > 0) else "failed"
            
            result = ProcessingResult(
                doc_id=doc_id,
                text_chunks=text_chunks,
                image_count=image_count,
                processing_time=processing_time,
                doc_type=doc_type,
                status=status,
                errors=errors if errors else None
            )
            
            self.logger.info(f"Unified processing completed: {result}")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unified processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                doc_id=doc_id,
                text_chunks=0,
                image_count=0,
                processing_time=processing_time,
                status="failed",
                errors=[error_msg]
            )
    
    async def query(self, 
                   query_text: str, 
                   top_k: int = 5, 
                   mode: str = "auto",
                   max_tokens: int = 1000,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        Query the unified vector store for relevant content with smart mode selection
        
        Args:
            query_text: The search query
            top_k: Number of results to retrieve
            mode: Query mode ("auto", "hybrid", "text_only", "image_only")
            max_tokens: Maximum tokens for LLM response
            temperature: Temperature for LLM response
            
        Returns:
            Dictionary with response and metadata
        """
        # Check if this is a direct visual question that should use VQA
        if self.vqa_engine and mode in ["auto", "image_only", "hybrid"]:
            is_visual_question = await self.vqa_engine.is_visual_question(query_text)
            
            if is_visual_question:
                self.logger.info(f"Detected visual question, using VQA: {query_text}")
                vqa_result = await self.vqa_engine.answer_visual_question(query_text, top_k)
                
                if vqa_result["status"] == "success":
                    return {
                        "status": "success",
                        "response": vqa_result["answer"],
                        "sources": vqa_result.get("image_responses", []),
                        "context_used": vqa_result["images_analyzed"],
                        "text_sources": 0,
                        "image_sources": vqa_result["images_analyzed"],
                        "query_mode": "visual_qa",
                        "confidence": vqa_result.get("confidence", 0.8)
                    }
                elif vqa_result["status"] == "no_images":
                    # Fall through to regular processing if no images found
                    self.logger.info("VQA found no relevant images, falling back to regular processing")
                else:
                    # VQA failed, fall through to regular processing
                    self.logger.warning(f"VQA failed: {vqa_result.get('answer', 'Unknown error')}")
        
        # Auto-detect optimal query mode if mode is "auto" and not handled by VQA
        if mode == "auto":
            mode = self._determine_optimal_query_mode(query_text)
            self.logger.info(f"Auto-detected query mode: {mode}")
        
        self.logger.info(f"Processing unified query: {query_text} (mode: {mode})")
        
        try:
            # 1. Generate embedding for query
            query_embeddings = await self.embedding_func([query_text])
            query_embedding = query_embeddings[0]
            
            # 2. Search vector store
            raw_results = self.vector_store.search(query_embedding, k=top_k)
            
            # 3. Filter results based on mode
            filtered_results = self._filter_results_by_mode(raw_results, mode)
            
            if not filtered_results:
                return {
                    "status": "success",
                    "response": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "context_used": 0,
                    "text_sources": 0,
                    "image_sources": 0
                }
            
            # 4. Prepare context from results
            context = self._prepare_context(filtered_results)
            
            # 5. Create comprehensive prompt
            prompt = self._create_query_prompt(query_text, context, mode)
            
            # 6. Generate response with LLM
            response = await self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 7. Prepare response metadata
            text_sources = sum(1 for r in filtered_results if r.get('metadata', {}).get('content_type') == 'text')
            image_sources = sum(1 for r in filtered_results if r.get('metadata', {}).get('content_type') == 'image')
            
            return {
                "status": "success",
                "response": response,
                "sources": [r.get('metadata', {}) for r in filtered_results],
                "context_used": len(filtered_results),
                "text_sources": text_sources,
                "image_sources": image_sources,
                "query_mode": mode
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": None,
                "sources": [],
                "context_used": 0
            }
    
    def _filter_results_by_mode(self, results: List[Dict], mode: str) -> List[Dict]:
        """Filter search results based on query mode"""
        if mode == "text_only":
            return [r for r in results if r.get('metadata', {}).get('content_type') == 'text']
        elif mode == "image_only":
            return [r for r in results if r.get('metadata', {}).get('content_type') == 'image']
        else:  # hybrid mode
            return results
    
    def _prepare_context(self, results: List[Dict]) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            content_type = result.get('metadata', {}).get('content_type', 'unknown')
            text_content = result.get('text', '')
            
            if content_type == "text":
                context_parts.append(f"[Document Text {i}]:\n{text_content}")
            elif content_type == "image":
                image_filename = result.get('metadata', {}).get('image_filename', 'unknown')
                context_parts.append(f"[Image Description {i}] (File: {image_filename}):\n{text_content}")
            else:
                context_parts.append(f"[Content {i}]:\n{text_content}")
        
        return "\n\n".join(context_parts)
    
    def _create_query_prompt(self, query_text: str, context: str, mode: str) -> str:
        """Create a comprehensive prompt for the LLM"""
        mode_descriptions = {
            "hybrid": "both text content and visual elements (images/diagrams)",
            "text_only": "text content only",
            "image_only": "visual elements (images/diagrams) only"
        }
        
        mode_desc = mode_descriptions.get(mode, "available content")
        
        prompt = f"""Based on the provided context from a document that includes {mode_desc}, please answer the following question comprehensively.

Context:
{context}

Question: {query_text}

Instructions:
- Provide a detailed and accurate answer based on the context provided
- If the context includes both text and image descriptions, integrate information from both sources
- Clearly distinguish between information from text content and visual elements when relevant
- If you cannot answer the question based on the provided context, say so clearly
- Be specific and cite relevant details from the context

Answer:"""
        
        return prompt
    
    async def get_document_summary(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Get a summary of all processed documents in the vector store
        
        Args:
            top_k: Number of content pieces to analyze
            
        Returns:
            Summary information about the document collection
        """
        try:
            # Get a sample of stored content
            sample_query = "document content summary overview"
            query_embeddings = await self.embedding_func([sample_query])
            query_embedding = query_embeddings[0]
            
            results = self.vector_store.search(query_embedding, k=top_k)
            
            # Analyze content
            text_count = sum(1 for r in results if r.get('metadata', {}).get('content_type') == 'text')
            image_count = sum(1 for r in results if r.get('metadata', {}).get('content_type') == 'image')
            
            # Get unique source documents
            sources = set()
            for r in results:
                source = r.get('metadata', {}).get('source', 'unknown')
                if source != 'unknown':
                    sources.add(source)
            
            return {
                "status": "success",
                "total_content_pieces": len(results),
                "text_chunks": text_count,
                "image_descriptions": image_count,
                "unique_documents": len(sources),
                "document_sources": list(sources)
            }
            
        except Exception as e:
            self.logger.error(f"Document summary failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _determine_document_type(self, text_chunks: int, image_count: int) -> str:
        """
        Determine document type based on processed content
        
        Args:
            text_chunks: Number of text chunks processed
            image_count: Number of images processed
            
        Returns:
            Document type: 'image_only', 'text_only', 'multimodal', or 'empty'
        """
        if text_chunks == 0 and image_count == 0:
            return "empty"
        elif text_chunks == 0 and image_count > 0:
            return "image_only"
        elif text_chunks > 0 and image_count == 0:
            return "text_only"
        else:
            return "multimodal"

    def _determine_optimal_query_mode(self, query_text: str) -> str:
        """
        Automatically determine the best query mode based on query content and document types
        
        Args:
            query_text: User's query
            
        Returns:
            Optimal query mode: "hybrid", "text_only", "image_only"
        """
        query_lower = query_text.lower()
        
        # Visual question indicators
        visual_indicators = [
            "image", "picture", "diagram", "chart", "figure", "graph", "visual", 
            "show", "display", "illustrate", "color", "shape", "appearance",
            "in the image", "what does the diagram", "what is shown", "what's shown",
            "solar system", "sun", "planet", "celestial", "astronomical"
        ]
        
        # Text-specific indicators
        text_indicators = [
            "definition", "explain the text", "what does the document say",
            "paragraph", "section", "chapter", "written", "text content"
        ]
        
        # Check for visual indicators
        visual_score = sum(2 if indicator in query_lower else 0 for indicator in visual_indicators)
        text_score = sum(1 if indicator in query_lower else 0 for indicator in text_indicators)
        
        # Check document types in our corpus
        doc_types = []
        for doc_metadata in self.document_metadata.values():
            doc_types.append(doc_metadata.get("doc_type", "unknown"))
        
        # Make decision based on indicators and document types
        if visual_score >= 3:  # Strong visual indicators
            if all(dtype == "image_only" for dtype in doc_types):
                return "image_only"
            else:
                return "hybrid"  # Mix of visual query with potentially mixed content
        
        elif text_score >= 2:  # Strong text indicators
            return "text_only"
        
        elif not doc_types:  # No documents processed yet
            return "hybrid"
        
        elif all(dtype == "image_only" for dtype in doc_types):
            # All documents are image-only, default to image_only
            return "image_only"
        
        elif all(dtype == "text_only" for dtype in doc_types):
            # All documents are text-only, default to text_only
            return "text_only"
        
        else:
            # Mixed document types or unknown - use hybrid
            return "hybrid"

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a processed document"""
        return self.document_metadata.get(doc_id, {
            "doc_id": doc_id,
            "processed_at": datetime.now().isoformat()
        })
        
    def get_all_document_metadata(self) -> Dict[str, Any]:
        """Get metadata for all processed documents"""
        return self.document_metadata.copy()


def create_unified_pdf_processor(api_key: str,
                               use_mock: bool = False,
                               embedding_model: str = "all-MiniLM-L6-v2",
                               collection_name: str = "unified_multimodal_documents",
                               enable_vqa: bool = True) -> UnifiedPDFProcessor:
    """
    Factory function to create a unified PDF processor with VQA support
    
    Args:
        api_key: DeepSeek API key
        use_mock: Whether to use mock clients for testing
        embedding_model: Name of the embedding model
        collection_name: Name of the vector database collection
        enable_vqa: Whether to enable Visual Question Answering capabilities
        
    Returns:
        Configured UnifiedPDFProcessor instance with VQA support
    """
    # Create LLM client
    if use_mock:
        llm_client = MockLLMClient()
    else:
        llm_client = MockLLMClient()
    
    # Create text processor
    text_processor, text_query_engine = create_pdf_processor(
        model_name=embedding_model,
        collection_name=collection_name
    )
    
    # Create vision model function
    async def vision_model_func(prompt: str, system_prompt: str, image_data: str) -> str:
        return await llm_client.describe_image(prompt, system_prompt, image_data)
    
    # Create image repository for VQA (if enabled)
    image_repository = ImageRepository() if enable_vqa else None
    
    # Create image processor (reusing the same vector store)
    image_processor, image_query_engine = create_pdf_image_processor(
        vision_model_func=vision_model_func,
        embedding_func=text_processor.embedding_func,
        vector_store=text_processor.vector_store,
        image_repository=image_repository
    )
    
    # Create unified processor with VQA support
    unified_processor = UnifiedPDFProcessor(
        llm_client=llm_client,
        vision_model_func=vision_model_func,
        embedding_func=text_processor.embedding_func,
        vector_store=text_processor.vector_store,
        text_processor=text_processor,
        image_processor=image_processor,
        image_repository=image_repository,
        enable_vqa=enable_vqa
    )
    
    return unified_processor


# Test function for validation
async def test_unified_processor():
    """Test the unified PDF processor"""
    
    # Create unified processor with mock mode
    processor = create_unified_pdf_processor(
        api_key=None,
        use_mock=True,
        collection_name="test_unified_multimodal"
    )
    
    # Test with sample PDF
    test_pdf = "DemoData/multimodal_test.pdf"
    
    if os.path.exists(test_pdf):
        print("Testing unified PDF processing...")
        
        # Process PDF
        result = await processor.process_pdf_complete(
            pdf_path=test_pdf,
            process_text=True,
            process_images=True,
            chunk_size=500,
            chunk_overlap=100
        )
        
        print(f"Processing result: {result}")
        
        # Test different query modes
        test_queries = [
            ("What is machine learning?", "text_only"),
            ("Describe the diagrams in the document", "image_only"),
            ("Explain the machine learning workflow with diagrams", "hybrid")
        ]
        
        for query, mode in test_queries:
            print(f"\nTesting {mode} query: {query}")
            response = await processor.query(query, mode=mode, top_k=5)
            print(f"Response: {response.get('response', 'No response')[:200]}...")
            print(f"Sources: {response.get('text_sources', 0)} text, {response.get('image_sources', 0)} images")
        
        # Get document summary
        summary = await processor.get_document_summary()
        print(f"\nDocument summary: {summary}")
        
    else:
        print(f"Test PDF not found: {test_pdf}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_unified_processor())