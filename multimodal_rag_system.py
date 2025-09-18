"""
Multimodal RAG System - Combining Text and Image Processing

This module implements a unified multimodal RAG system that can process both text
and images from PDF documents for comprehensive retrieval-augmented generation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pdf_text_processor import SimplePDFTextProcessor, PDFQueryEngine, create_pdf_processor
from pdf_image_processor import SimplePDFImageProcessor, PDFImageQueryEngine, create_pdf_image_processor


class MultimodalRAGSystem:
    """
    Unified multimodal RAG system that processes both text and images from PDFs
    """
    
    def __init__(self, 
                 text_processor: SimplePDFTextProcessor,
                 image_processor: SimplePDFImageProcessor,
                 text_query_engine: PDFQueryEngine,
                 image_query_engine: PDFImageQueryEngine,
                 llm_client,
                 max_context_chunks: int = 5):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.text_query_engine = text_query_engine
        self.image_query_engine = image_query_engine
        self.llm_client = llm_client
        self.max_context_chunks = max_context_chunks
        self.logger = logging.getLogger(__name__)
    
    async def add_document(self, pdf_path: str, 
                          process_text: bool = True, 
                          process_images: bool = True,
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 200,
                          image_output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a PDF document to the multimodal RAG system
        
        Args:
            pdf_path: Path to the PDF file
            process_text: Whether to process text content
            process_images: Whether to process image content
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            image_output_dir: Directory to save extracted images
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Adding multimodal document: {pdf_path}")
        
        results = {
            "status": "success",
            "pdf_path": pdf_path,
            "text_result": None,
            "image_result": None
        }
        
        try:
            # Process text content
            if process_text:
                self.logger.info("Processing text content...")
                text_result = await self.text_processor.process_pdf_text(
                    pdf_path, chunk_size, chunk_overlap
                )
                results["text_result"] = {
                    "chunks_processed": text_result,
                    "status": "success"
                }
                self.logger.info(f"Text processing completed: {text_result} chunks")
            
            # Process image content
            if process_images:
                self.logger.info("Processing image content...")
                image_result = await self.image_processor.process_pdf_images(
                    pdf_path, image_output_dir
                )
                results["image_result"] = image_result
                self.logger.info(f"Image processing completed: {image_result.get('images_processed', 0)} images")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing multimodal document: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "pdf_path": pdf_path,
                "text_result": None,
                "image_result": None
            }
    
    async def query(self, query: str, 
                   search_text: bool = True,
                   search_images: bool = True,
                   max_tokens: int = 1000,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        Query the multimodal RAG system
        
        Args:
            query: Search query
            search_text: Whether to search text content
            search_images: Whether to search image content
            max_tokens: Maximum tokens for LLM response
            temperature: Temperature for LLM response
            
        Returns:
            Dictionary with query results
        """
        self.logger.info(f"Processing multimodal query: {query}")
        
        try:
            context_parts = []
            text_results = []
            image_results = []
            
            # Search text content
            if search_text:
                text_results = await self.text_query_engine.query(
                    query, k=self.max_context_chunks
                )
                if text_results:  # text_query_engine returns a list directly
                    for i, result in enumerate(text_results):
                        context_parts.append(f"Text Context {i+1}:\n{result['text']}\n")
            
            # Search image content
            if search_images:
                image_query_result = await self.image_query_engine.query(
                    query, k=self.max_context_chunks
                )
                if image_query_result.get("status") == "success" and image_query_result.get("results"):
                    image_results = image_query_result["results"]
                    for i, result in enumerate(image_results):
                        context_parts.append(f"Image Context {i+1}:\n{result['text']}\n")
            
            # Combine contexts
            if not context_parts:
                return {
                    "status": "success",
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "context": [],
                    "context_chunks": 0,
                    "text_chunks": 0,
                    "image_chunks": 0
                }
            
            combined_context = "\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""Based on the following context from a document, please answer the question.

Context:
{combined_context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context contains both text and image descriptions, integrate information from both sources in your response."""
            
            # Generate response
            response = await self.llm_client.generate_response(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            
            # Combine all results for context
            all_results = []
            if text_results:
                all_results.extend(text_results)
            if image_results:
                all_results.extend(image_results)
            
            return {
                "status": "success",
                "answer": response,
                "context": all_results,
                "context_chunks": len(context_parts),
                "text_chunks": len(text_results) if text_results else 0,
                "image_chunks": len(image_results) if image_results else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in multimodal query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "answer": None,
                "context": [],
                "context_chunks": 0
            }


def create_multimodal_rag_system(api_key: str, 
                                use_mock: bool = False,
                                embedding_model: str = "all-MiniLM-L6-v2",
                                collection_name: str = "multimodal_rag_documents") -> MultimodalRAGSystem:
    """
    Factory function to create a complete multimodal RAG system
    
    Args:
        api_key: API key (None for mock mode)
        use_mock: Whether to use mock clients for testing
        embedding_model: Name of the embedding model
        collection_name: Name of the vector database collection
        
    Returns:
        Configured MultimodalRAGSystem instance
    """
    from rag_core import MockLLMClient
    
    # Create LLM client
    if use_mock:
        llm_client = MockLLMClient()
    else:
        llm_client = MockLLMClient()
    
    # Create text processor
    text_processor, text_query_engine = create_pdf_processor(
        model_name=embedding_model,
        collection_name=f"{collection_name}_text"
    )
    
    # Create vision model function
    async def vision_model_func(prompt: str, system_prompt: str, image_data: str) -> str:
        return await llm_client.describe_image(prompt, system_prompt, image_data)
    
    # Create image processor
    image_processor, image_query_engine = create_pdf_image_processor(
        vision_model_func=vision_model_func,
        embedding_func=text_processor.embedding_func,  # Reuse the same embedding function
        vector_store=text_processor.vector_store  # Reuse the same vector store for unified search
    )
    
    # Create multimodal system
    system = MultimodalRAGSystem(
        text_processor=text_processor,
        image_processor=image_processor,
        text_query_engine=text_query_engine,
        image_query_engine=image_query_engine,
        llm_client=llm_client,
        max_context_chunks=5
    )
    
    return system


# Test function
async def test_multimodal_system():
    """Test function for the multimodal RAG system"""
    import os
    
    # Create system with mock mode
    system = create_multimodal_rag_system(
        api_key=None,
        use_mock=True,  # Use mock for testing
        collection_name="test_multimodal"
    )
    
    # Test document processing
    if os.path.exists(PDF_PATH):
        print("Processing PDF document...")
        result = await system.add_document(
            PDF_PATH,
            process_text=True,
            process_images=True
        )
        print(f"Processing result: {result}")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Describe any diagrams or images in the document",
            "What visual elements are present in the document?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await system.query(
                query, 
                search_text=True, 
                search_images=True
            )
            print(f"Answer: {response.get('answer', 'No answer')[:200]}...")
            print(f"Context chunks: {response.get('context_chunks', 0)} (Text: {response.get('text_chunks', 0)}, Images: {response.get('image_chunks', 0)})")
    else:
        print(f"PDF file not found: {PDF_PATH}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_multimodal_system())