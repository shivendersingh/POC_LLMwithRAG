"""
Visual Question Answering (VQA) Engine

This module provides direct visual question answering capabilities,
allowing users to ask specific questions about images in documents.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from image_repository import ImageRepository


class VisualQAEngine:
    """
    Engine for direct visual question answering
    
    Provides capability to ask specific questions about images,
    going beyond pre-generated descriptions.
    """
    
    def __init__(self, 
                 vision_model_func, 
                 image_repository: ImageRepository, 
                 vector_store,
                 embedding_func):
        self.vision_model_func = vision_model_func
        self.image_repository = image_repository
        self.vector_store = vector_store
        self.embedding_func = embedding_func
        self.logger = logging.getLogger(__name__)
        
    async def is_visual_question(self, query_text: str) -> bool:
        """
        Determine if this is a visual-specific question
        
        Args:
            query_text: User's question
            
        Returns:
            True if this appears to be a visual question
        """
        visual_indicators = [
            "in the image", "in the diagram", "in the picture", "in the figure",
            "what does the image", "what does the diagram", "what does the picture",
            "show me", "can you see", "what's shown", "what is shown", 
            "visual", "look like", "appears", "displayed", "depicted",
            "color", "shape", "appearance", "diagram shows", "image display",
            "what color", "how many", "where is", "what type of"
        ]
        
        query_lower = query_text.lower()
        
        # Count visual indicators
        indicator_score = sum(2 if indicator in query_lower else 0 for indicator in visual_indicators)
        
        # Check for visual question patterns
        pattern_score = 0
        if re.search(r'what (is|are) .* (showing|displaying|depicted)', query_lower):
            pattern_score += 3
        if re.search(r'how (does|do) .* look', query_lower):
            pattern_score += 3
        if re.search(r'what (can you see|do you see)', query_lower):
            pattern_score += 3
        if re.search(r'describe .* (image|diagram|picture|figure)', query_lower):
            pattern_score += 2
            
        total_score = indicator_score + pattern_score
        
        # Log detection results
        if total_score >= 2:
            self.logger.info(f"Visual question detected (score: {total_score}): {query_text}")
            return True
        else:
            self.logger.debug(f"Not a visual question (score: {total_score}): {query_text}")
            return False
            
    async def answer_visual_question(self, 
                                   query_text: str, 
                                   top_k: int = 3,
                                   doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question about images directly using VQA
        
        Args:
            query_text: The visual question to answer
            top_k: Maximum number of images to analyze
            doc_id: Optional specific document ID to search within
            
        Returns:
            Dictionary with VQA results
        """
        self.logger.info(f"Processing visual question: {query_text}")
        
        try:
            # 1. Find relevant images based on text similarity
            relevant_images = await self._find_relevant_images(query_text, top_k, doc_id)
            
            if not relevant_images:
                return {
                    "status": "no_images",
                    "answer": "I couldn't find any relevant images to answer your question.",
                    "images_analyzed": 0,
                    "confidence": 0.0
                }
            
            # 2. For each relevant image, directly ask the vision model the specific question
            responses = []
            for img_data in relevant_images:
                try:
                    response = await self.vision_model_func(
                        prompt=self._create_vqa_prompt(query_text),
                        system_prompt=self._create_vqa_system_prompt(),
                        image_data=img_data["image_data"]
                    )
                    
                    responses.append({
                        "answer": response,
                        "metadata": img_data["metadata"],
                        "doc_id": img_data["doc_id"],
                        "image_id": img_data["image_id"]
                    })
                    
                    self.logger.info(f"VQA response generated for image {img_data['image_id']}")
                    
                except Exception as e:
                    self.logger.error(f"VQA failed for image {img_data.get('image_id', 'unknown')}: {str(e)}")
                    continue
            
            if not responses:
                return {
                    "status": "processing_failed",
                    "answer": "I found relevant images but couldn't analyze them due to technical issues.",
                    "images_analyzed": 0,
                    "confidence": 0.0
                }
            
            # 3. Combine and format responses
            final_answer = self._combine_vqa_responses(responses, query_text)
            
            return {
                "status": "success",
                "answer": final_answer,
                "images_analyzed": len(responses),
                "confidence": self._calculate_confidence(responses),
                "image_responses": responses,
                "query_mode": "visual_qa"
            }
            
        except Exception as e:
            self.logger.error(f"VQA processing failed: {str(e)}")
            return {
                "status": "error",
                "answer": f"I encountered an error while processing your visual question: {str(e)}",
                "images_analyzed": 0,
                "confidence": 0.0
            }
    
    async def _find_relevant_images(self, 
                                  query_text: str, 
                                  top_k: int,
                                  doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find images relevant to the query"""
        try:
            # If doc_id is specified, get all images from that document
            if doc_id:
                images = self.image_repository.get_all_images_for_document(doc_id)
                return images[:top_k]  # Limit to top_k
            
            # Otherwise, use vector similarity search
            query_embeddings = await self.embedding_func([query_text])
            query_embedding = query_embeddings[0]
            
            # Search for image content in vector store
            results = self.vector_store.search(
                query_embedding, 
                k=top_k * 2,  # Get more results to filter
                filter_dict={"content_type": "image"}
            )
            
            if not results:
                return []
            
            # Get embedding IDs from search results
            embedding_ids = []
            for result in results:
                metadata = result.get('metadata', {})
                embedding_id = metadata.get('embedding_id')
                if embedding_id:
                    embedding_ids.append(embedding_id)
            
            # Get images from repository
            images = self.image_repository.get_images_by_embeddings(embedding_ids)
            
            return images[:top_k]  # Limit to requested number
            
        except Exception as e:
            self.logger.error(f"Failed to find relevant images: {str(e)}")
            return []
    
    def _create_vqa_prompt(self, query_text: str) -> str:
        """Create an optimized prompt for visual question answering"""
        return f"""Please answer this specific question about the image: {query_text}

Instructions:
- Focus exclusively on what you can see in the image
- Be specific and detailed in your observations
- If the question asks about text in the image, read and report it accurately
- If the question asks about colors, shapes, or spatial relationships, describe them precisely
- If you cannot answer based on the visible content, clearly state what cannot be determined
- Provide context for your observations when relevant

Question: {query_text}"""
    
    def _create_vqa_system_prompt(self) -> str:
        """Create system prompt for VQA"""
        return """You are an expert Visual Question Answering AI assistant. 
        
Your task is to analyze the provided image and answer the specific question about it accurately and in detail.

Guidelines:
- Examine the image carefully and provide precise, factual answers
- Focus on visual elements: objects, colors, shapes, text, spatial relationships, quantities
- If the image contains diagrams, charts, or technical content, explain what they represent
- If there's text in the image, read it accurately and incorporate it into your answer
- Be honest about limitations - if something cannot be determined from the image alone, say so
- Provide comprehensive answers that fully address the question asked
- Use clear, descriptive language that would be helpful for someone who cannot see the image"""
    
    def _combine_vqa_responses(self, responses: List[Dict[str, Any]], query_text: str) -> str:
        """Combine multiple VQA responses into a coherent answer"""
        if len(responses) == 1:
            return responses[0]["answer"]
        
        # Multiple images - provide structured response
        combined = f"Based on my analysis of {len(responses)} relevant images:\n\n"
        
        for i, resp in enumerate(responses, 1):
            image_id = resp.get("image_id", f"Image {i}")
            answer = resp["answer"]
            combined += f"**Image {i} ({image_id}):**\n{answer}\n\n"
        
        # Add summary if multiple images
        if len(responses) > 1:
            combined += "**Summary:**\n"
            combined += f"I analyzed {len(responses)} images to answer your question about '{query_text}'. "
            combined += "Each image provided different visual information relevant to your query."
        
        return combined
    
    def _calculate_confidence(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for VQA responses"""
        if not responses:
            return 0.0
        
        # Simple confidence calculation based on response quality
        base_confidence = 0.8  # Base confidence for successful VQA
        
        # Boost confidence if multiple images confirm findings
        if len(responses) > 1:
            base_confidence += 0.1
        
        # Check response quality indicators
        total_length = sum(len(resp["answer"]) for resp in responses)
        avg_length = total_length / len(responses)
        
        # Longer, more detailed responses generally indicate higher confidence
        if avg_length > 200:
            base_confidence += 0.05
        elif avg_length < 50:
            base_confidence -= 0.1
        
        return min(base_confidence, 1.0)
    
    def get_vqa_stats(self) -> Dict[str, Any]:
        """Get VQA engine statistics"""
        repo_stats = self.image_repository.get_storage_stats()
        
        return {
            "total_images_available": repo_stats["total_images"],
            "total_documents": repo_stats["total_documents"],
            "storage_size_mb": repo_stats["storage_size_mb"],
            "vqa_capabilities": [
                "Direct image questioning",
                "Multi-image analysis",
                "OCR text reading",
                "Visual element description",
                "Spatial relationship analysis"
            ]
        }


# Test function
async def test_vqa_engine():
    """Test the VQA engine functionality"""
    from image_repository import ImageRepository
    from rag_core import MockLLMClient
    
    # Create mock components
    repo = ImageRepository("./test_vqa_storage")
    mock_client = MockLLMClient()
    
    # Mock vision function
    async def mock_vision_func(prompt: str, system_prompt: str, image_data: str) -> str:
        return f"Mock VQA response for prompt: {prompt[:50]}..."
    
    # Mock embedding function
    async def mock_embedding_func(texts: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in texts]
    
    # Mock vector store
    class MockVectorStore:
        def search(self, embedding, k=5, filter_dict=None):
            return [{"metadata": {"embedding_id": "test_emb_1", "content_type": "image"}}]
    
    vector_store = MockVectorStore()
    
    # Create VQA engine
    vqa_engine = VisualQAEngine(
        vision_model_func=mock_vision_func,
        image_repository=repo,
        vector_store=vector_store,
        embedding_func=mock_embedding_func
    )
    
    # Test visual question detection
    test_questions = [
        "What is shown in the diagram?",
        "What color is the sun in the image?",
        "Explain the concept of machine learning",  # Not visual
        "How many planets are visible?"
    ]
    
    for question in test_questions:
        is_visual = await vqa_engine.is_visual_question(question)
        print(f"'{question}' -> Visual: {is_visual}")
    
    # Test VQA (would need actual images for full test)
    stats = vqa_engine.get_vqa_stats()
    print(f"VQA Stats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vqa_engine())