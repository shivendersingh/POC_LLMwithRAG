"""
RAG Core System - Local Processing Only

This module provides the core RAG (Retrieval-Augmented Generation) functionality
for PDF text processing with local mock responses (no API keys required).
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from pdf_text_processor import SimplePDFTextProcessor, PDFQueryEngine, create_pdf_processor


class MockLLMClient:
    """Mock client for local processing without API calls"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a mock response for testing"""
        self.logger.info("Using mock LLM client")

        # Simple mock response based on context
        if "context" in prompt.lower():
            return f"""Based on the provided context, I can help answer your question.

This is a mock response for testing purposes. The actual LLM would provide
a more sophisticated answer based on the retrieved document content.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            return "This is a mock response from the LLM client for testing purposes."

    async def describe_image(self, prompt: str, system_prompt: str, image_data: str) -> str:
        """Generate a mock image description for testing"""
        self.logger.info("Using mock vision client")

        # Mock description based on image data size and prompt
        image_size = len(image_data)
        return f"""Mock image description: This appears to be a document image with {image_size} base64 characters.
The image contains visual elements that would be relevant for document analysis and search.
This is a mock description generated for testing purposes.

Prompt used: {prompt[:50]}...
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) System
    
    Combines PDF text processing, vector search, and LLM
    for intelligent document-based question answering.
    """
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Sending image description request to DeepSeek Vision API (attempt {attempt + 1})")
                
                # Use asyncio to run the synchronous request in a thread
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=90  # Increased timeout to 90 seconds
                    )
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    self.logger.info("Successfully received image description from DeepSeek Vision")
                    return content
                else:
                    error_text = response.text
                    self.logger.warning(f"DeepSeek Vision API returned {response.status_code} on attempt {attempt + 1}: {error_text}")
                    if attempt == max_retries - 1:
                        # Last attempt failed, use fallback
                        self.logger.warning("All API attempts failed, falling back to enhanced description")
                        return await self._fallback_image_description(prompt, system_prompt, image_data)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    self.logger.warning("All attempts timed out, using fallback")
                    return await self._fallback_image_description(prompt, system_prompt, image_data)
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.warning(f"DeepSeek Vision API failed on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.warning("All attempts failed, using fallback")
                    return await self._fallback_image_description(prompt, system_prompt, image_data)
                await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries failed
        return await self._fallback_image_description(prompt, system_prompt, image_data)
    
    async def _fallback_image_description(self, prompt: str, system_prompt: str, image_data: str) -> str:
        """Enhanced fallback image description with intelligent analysis"""
        try:
            # Try to analyze image metadata and colors
            image_analysis = self._analyze_image_metadata(image_data)
            
            fallback_prompt = f"""
            {system_prompt}
            
            I need to analyze an image from a PDF document. Based on the image metadata analysis:
            - Image size: {image_analysis['size_category']}
            - Estimated content type: {image_analysis['content_type']}
            - Detected characteristics: {image_analysis['characteristics']}
            
            This appears to be from a PDF document, likely containing:
            {image_analysis['likely_content']}
            
            User request: {prompt}
            
            Please provide a comprehensive description that would help answer questions about:
            1. The type of visual content (diagram, chart, illustration, etc.)
            2. Likely subject matter based on common PDF content patterns
            3. How this visual element would relate to document analysis
            4. Potential information that could be extracted from such content
            
            Focus on creating a description that would be useful for RAG retrieval and question answering.
            If this appears to be an astronomical or scientific diagram, provide relevant context.
            """
            
            response = await self.generate_response(fallback_prompt, max_tokens=500, temperature=0.3)
            
            # Enhance response with image analysis
            enhanced_response = f"{response}\n\nTechnical Analysis: {image_analysis['technical_details']}"
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Enhanced fallback description failed: {str(e)}")
            
            # Basic fallback with image size analysis
            size_category = "large" if len(image_data) > 100000 else "medium" if len(image_data) > 20000 else "small"
            
            return f"""Image Analysis (Enhanced Fallback): Document image detected ({size_category} size, {len(image_data)} base64 characters).
            
            This appears to be a visual element from a PDF document that likely contains:
            - Scientific or educational content (based on common PDF patterns)
            - Diagrams, charts, or illustrations
            - Potentially astronomical, technical, or academic subject matter
            - Visual information that complements the document text

            Content Analysis:
            - Image complexity: {size_category} (based on file size)
            - Likely purpose: Illustrative or explanatory content
            - Context relevance: High (embedded in PDF document)

            User Query Context: {prompt[:100]}...
            Processing Note: Vision API unavailable, using intelligent fallback analysis
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Recommendation: This image may contain important visual information relevant to your query.
            Consider the context of astronomical diagrams, scientific illustrations, or educational content.
            """

    def _analyze_image_metadata(self, image_data: str) -> Dict[str, str]:
        """Analyze image metadata to provide better fallback descriptions"""
        try:
            import base64
            from PIL import Image
            import io

            # Decode and analyze image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            format_type = img.format
            mode = img.mode

            # Analyze image characteristics
            size_category = "large" if len(image_data) > 100000 else "medium" if len(image_data) > 20000 else "small"
            aspect_ratio = width / height if height > 0 else 1

            # Determine likely content type
            if aspect_ratio > 1.5:
                content_type = "wide diagram or chart"
            elif aspect_ratio < 0.7:
                content_type = "tall diagram or figure"
            else:
                content_type = "square or standard diagram"

            # Analyze colors (if possible)
            characteristics = []
            try:
                # Get dominant colors
                colors = img.getcolors(maxcolors=1000)
                if colors:
                    colors.sort(reverse=True, key=lambda x: x[0])
                    for count, color in colors[:3]:
                        if isinstance(color, tuple) and len(color) >= 3:
                            r, g, b = color[:3]
                            if r > 200 and g < 100 and b < 100:
                                characteristics.append("contains red elements")
                            elif r < 100 and g < 100 and b > 200:
                                characteristics.append("contains blue elements")
                            elif r > 200 and g > 200 and b < 100:
                                characteristics.append("contains yellow elements")
                            elif r < 100 and g < 100 and b < 100:
                                characteristics.append("contains dark/black elements")
            except:
                characteristics.append("color analysis unavailable")
            
            # Determine likely content based on characteristics
            if "red elements" in characteristics and "yellow elements" in characteristics:
                likely_content = "Potentially an astronomical diagram (solar system, planetary illustration)"
            elif size_category == "large" and aspect_ratio < 1.2:
                likely_content = "Detailed diagram or complex illustration"
            else:
                likely_content = "Scientific or educational illustration"
            
            return {
                "size_category": size_category,
                "content_type": content_type,
                "characteristics": ", ".join(characteristics) if characteristics else "standard image characteristics",
                "likely_content": likely_content,
                "technical_details": f"Dimensions: {width}x{height}, Format: {format_type}, Mode: {mode}, Aspect Ratio: {aspect_ratio:.2f}"
            }
            
        except Exception as e:
            self.logger.warning(f"Image metadata analysis failed: {str(e)}")
            return {
                "size_category": "unknown",
                "content_type": "standard image",
                "characteristics": "analysis unavailable",
                "likely_content": "visual content from PDF document",
                "technical_details": f"Size: {len(image_data)} base64 characters"
            }


class RAGSystem:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a mock response for testing"""
        self.logger.info("Using mock DeepSeek client")
        
        # Simple mock response based on context
        if "context" in prompt.lower():
            return f"""Based on the provided context, I can help answer your question. 
            This is a mock response for testing purposes. The actual DeepSeek API would provide 
            a more sophisticated answer based on the retrieved document content.
            
            Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        else:
            return "This is a mock response from the DeepSeek client for testing purposes."
    
    async def describe_image(self, prompt: str, system_prompt: str, image_data: str) -> str:
        """Generate a mock image description for testing"""
        self.logger.info("Using mock DeepSeek vision client")
        
        # Mock description based on image data size and prompt
        image_size = len(image_data)
        return f"""Mock image description: This appears to be a document image with {image_size} base64 characters.
        The image contains visual elements that would be relevant for document analysis and search.
        This is a mock description generated for testing purposes.
        
        Prompt used: {prompt[:50]}...
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """


class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) System
    
    Combines PDF text processing, vector search, and DeepSeek LLM
    for intelligent document-based question answering.
    """
    
    def __init__(self, 
                 pdf_processor: SimplePDFTextProcessor,
                 query_engine: PDFQueryEngine,
                 llm_client: MockLLMClient,
                 max_context_chunks: int = 5):
        self.pdf_processor = pdf_processor
        self.query_engine = query_engine
        self.llm_client = llm_client
        self.max_context_chunks = max_context_chunks
        self.logger = logging.getLogger(__name__)
    
    async def add_document(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Add a PDF document to the RAG system
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Processing results
        """
        self.logger.info(f"Adding document to RAG system: {pdf_path}")
        
        try:
            start_time = datetime.now()
            chunk_count = await self.pdf_processor.process_pdf_text(
                pdf_path, chunk_size, chunk_overlap
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "document_path": pdf_path,
                "chunks_processed": chunk_count,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully added document: {chunk_count} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "document_path": pdf_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error(f"Error adding document {pdf_path}: {str(e)}")
            return error_result
    
    async def query(self, question: str, max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        
        Args:
            question: User's question
            max_tokens: Maximum tokens for LLM response
            temperature: Temperature for LLM generation
            
        Returns:
            Complete response with context and answer
        """
        self.logger.info(f"Processing query: {question}")
        
        try:
            start_time = datetime.now()
            
            # Step 1: Retrieve relevant context
            context_results = await self.query_engine.query(question, k=self.max_context_chunks)
            
            # Step 2: Format context for LLM
            context_text = self._format_context(context_results)
            
            # Step 3: Create prompt for LLM
            prompt = self._create_rag_prompt(question, context_text)
            
            # Step 4: Generate response using LLM
            llm_response = await self.llm_client.generate_response(
                prompt, max_tokens, temperature
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare complete response
            response = {
                "status": "success",
                "question": question,
                "answer": llm_response,
                "context_chunks": len(context_results),
                "context": [
                    {
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                        "source": result["metadata"].get("source", "Unknown"),
                        "distance": result.get("distance"),
                        "chunk_id": result["metadata"].get("chunk_id")
                    }
                    for result in context_results
                ],
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully processed query in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            error_response = {
                "status": "error",
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.error(f"Error processing query: {str(e)}")
            return error_response
    
    def _format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """Format retrieved context for LLM prompt"""
        if not context_results:
            return "No relevant context found."
        
        formatted_contexts = []
        for i, result in enumerate(context_results, 1):
            source = result["metadata"].get("source", "Unknown")
            text = result["text"]
            formatted_contexts.append(f"Context {i} (from {source}):\n{text}\n")
        
        return "\n".join(formatted_contexts)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for the LLM"""
        prompt = f"""You are an AI assistant that answers questions based on provided context from documents.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based primarily on the provided context
- If the context doesn't contain enough information, clearly state this
- Be concise but comprehensive in your response
- Quote relevant parts of the context when appropriate
- If asked about something not in the context, explain that the information is not available in the provided documents

ANSWER:"""
        
        return prompt
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information"""
        return {
            "status": "operational",
            "max_context_chunks": self.max_context_chunks,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "pdf_processor": "operational",
                "query_engine": "operational", 
                "llm_client": "operational"
            }
        }


def create_rag_system(api_key: Optional[str] = None, 
                     use_mock_llm: bool = False,
                     model_name: str = "all-MiniLM-L6-v2",
                     collection_name: str = "pdf_documents") -> RAGSystem:
    """
    Factory function to create a complete RAG system
    
    Args:
        api_key: DeepSeek API key (None for mock)
        use_mock_llm: Whether to use mock LLM client
        model_name: Sentence transformer model name
        collection_name: ChromaDB collection name
        
    Returns:
        Configured RAG system
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create PDF processing components
    pdf_processor, query_engine = create_pdf_processor(model_name, collection_name)
    
    # Create LLM client
    if use_mock_llm or not api_key:
        llm_client = MockLLMClient()
    else:
        llm_client = MockLLMClient()
    
    # Create RAG system
    rag_system = RAGSystem(pdf_processor, query_engine, llm_client)
    
    return rag_system


if __name__ == "__main__":
    async def test_rag_system():
        """Test the complete RAG system"""
        # Create RAG system with mock LLM
        rag = create_rag_system(use_mock_llm=True)
        
        # Test system stats
        stats = await rag.get_system_stats()
        print("System Stats:")
        print(json.dumps(stats, indent=2))
        
        # Test adding a document (you'll need a valid PDF)
        pdf_path = "sample.pdf"  # Replace with actual PDF path
        
        try:
            print(f"\nAdding document: {pdf_path}")
            result = await rag.add_document(pdf_path)
            print("Add Document Result:")
            print(json.dumps(result, indent=2))
            
            # Test querying
            print("\nTesting query...")
            query_result = await rag.query("What is this document about?")
            print("Query Result:")
            print(json.dumps(query_result, indent=2))
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            print("Please provide a valid PDF file to test the complete system")
    
    # Run the test
    asyncio.run(test_rag_system())