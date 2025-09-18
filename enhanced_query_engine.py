"""
Enhanced Multimodal Query Engine with Advanced Context Handling

This module provides advanced query capabilities for the unified multimodal RAG system
with sophisticated context preparation and result ranking.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class QueryMode(Enum):
    """Query modes for multimodal search"""
    HYBRID = "hybrid"
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    AUTO = "auto"  # Automatically determine best mode


@dataclass
class ContextItem:
    """Individual context item with enhanced metadata"""
    content: str
    content_type: str  # 'text' or 'image'
    source_file: str
    chunk_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate content quality score"""
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate a quality score based on content characteristics"""
        score = self.similarity_score
        
        # Boost score for longer, more detailed content
        content_length = len(self.content)
        length_bonus = min(content_length / 1000, 0.2)  # Up to 0.2 bonus
        score += length_bonus
        
        # Boost score for image descriptions with specific details
        if self.content_type == 'image':
            detail_keywords = ['diagram', 'chart', 'figure', 'shows', 'displays', 'contains']
            detail_count = sum(1 for keyword in detail_keywords if keyword.lower() in self.content.lower())
            detail_bonus = min(detail_count * 0.05, 0.15)  # Up to 0.15 bonus
            score += detail_bonus
        
        return min(score, 1.0)  # Cap at 1.0


class EnhancedMultimodalQueryEngine:
    """
    Enhanced query engine with advanced context handling and result ranking
    """
    
    def __init__(self, unified_processor):
        self.unified_processor = unified_processor
        self.logger = logging.getLogger(__name__)
    
    async def advanced_query(self,
                           query_text: str,
                           mode: QueryMode = QueryMode.AUTO,
                           top_k: int = 8,
                           context_window: int = 4000,
                           enable_reranking: bool = True,
                           include_metadata: bool = True) -> Dict[str, Any]:
        """
        Advanced query with intelligent context selection and reranking
        
        Args:
            query_text: The search query
            mode: Query mode (auto, hybrid, text_only, image_only)
            top_k: Number of initial results to retrieve
            context_window: Maximum context window size in characters
            enable_reranking: Whether to enable intelligent reranking
            include_metadata: Whether to include detailed metadata
            
        Returns:
            Enhanced query results with advanced context
        """
        self.logger.info(f"Processing advanced query: {query_text}")
        
        try:
            # 1. Determine optimal query mode if AUTO
            if mode == QueryMode.AUTO:
                mode = await self._determine_optimal_mode(query_text)
                self.logger.info(f"Auto-determined mode: {mode.value}")
            
            # 2. Get initial results
            raw_results = await self._get_raw_results(query_text, top_k * 2)  # Get more for reranking
            
            # 3. Convert to enhanced context items
            context_items = self._create_context_items(raw_results)
            
            # 4. Filter by mode
            filtered_items = self._filter_by_mode(context_items, mode)
            
            # 5. Rerank if enabled
            if enable_reranking:
                filtered_items = await self._rerank_results(query_text, filtered_items)
            
            # 6. Select optimal context within window
            selected_context = self._select_optimal_context(filtered_items, context_window, top_k)
            
            # 7. Generate enhanced response
            response_data = await self._generate_enhanced_response(
                query_text, selected_context, mode, include_metadata
            )
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Advanced query failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": None
            }
    
    async def _determine_optimal_mode(self, query_text: str) -> QueryMode:
        """Automatically determine the best query mode based on query content"""
        query_lower = query_text.lower()
        
        # Keywords that suggest image/visual content
        visual_keywords = [
            'diagram', 'image', 'picture', 'chart', 'graph', 'figure', 'visual', 
            'illustration', 'drawing', 'photo', 'screenshot', 'show', 'display',
            'workflow', 'process', 'architecture', 'structure', 'layout'
        ]
        
        # Keywords that suggest text content
        text_keywords = [
            'definition', 'explain', 'describe', 'what is', 'how does', 'tell me about',
            'summary', 'details', 'information', 'content', 'text', 'document'
        ]
        
        visual_score = sum(1 for keyword in visual_keywords if keyword in query_lower)
        text_score = sum(1 for keyword in text_keywords if keyword in query_lower)
        
        # Determine mode based on keyword scores
        if visual_score > text_score and visual_score >= 2:
            return QueryMode.IMAGE_ONLY
        elif text_score > visual_score and text_score >= 2:
            return QueryMode.TEXT_ONLY
        else:
            return QueryMode.HYBRID  # Default to hybrid for balanced queries
    
    async def _get_raw_results(self, query_text: str, k: int) -> List[Dict]:
        """Get raw results from the vector store"""
        query_embeddings = await self.unified_processor.embedding_func([query_text])
        query_embedding = query_embeddings[0]
        return self.unified_processor.vector_store.search(query_embedding, k=k)
    
    def _create_context_items(self, raw_results: List[Dict]) -> List[ContextItem]:
        """Convert raw results to enhanced context items"""
        context_items = []
        
        for result in raw_results:
            metadata = result.get('metadata', {})
            
            item = ContextItem(
                content=result.get('text', ''),
                content_type=metadata.get('content_type', 'unknown'),
                source_file=metadata.get('source', 'unknown'),
                chunk_id=metadata.get('id', 'unknown'),
                similarity_score=1.0 - result.get('distance', 0.0),  # Convert distance to similarity
                metadata=metadata
            )
            
            context_items.append(item)
        
        return context_items
    
    def _filter_by_mode(self, items: List[ContextItem], mode: QueryMode) -> List[ContextItem]:
        """Filter context items by query mode"""
        if mode == QueryMode.TEXT_ONLY:
            return [item for item in items if item.content_type == 'text']
        elif mode == QueryMode.IMAGE_ONLY:
            return [item for item in items if item.content_type == 'image']
        else:  # HYBRID or AUTO
            return items
    
    async def _rerank_results(self, query_text: str, items: List[ContextItem]) -> List[ContextItem]:
        """Rerank results using quality scores and relevance"""
        # Sort by combined quality score and similarity
        def ranking_score(item: ContextItem) -> float:
            return 0.7 * item.similarity_score + 0.3 * item.quality_score
        
        reranked_items = sorted(items, key=ranking_score, reverse=True)
        
        self.logger.info(f"Reranked {len(items)} items")
        return reranked_items
    
    def _select_optimal_context(self, 
                              items: List[ContextItem], 
                              context_window: int, 
                              max_items: int) -> List[ContextItem]:
        """Select optimal context items within the context window"""
        selected_items = []
        current_length = 0
        
        for item in items[:max_items * 2]:  # Consider more items than max
            item_length = len(item.content) + 50  # Add overhead for formatting
            
            if current_length + item_length <= context_window and len(selected_items) < max_items:
                selected_items.append(item)
                current_length += item_length
            elif len(selected_items) >= max_items:
                break
        
        self.logger.info(f"Selected {len(selected_items)} context items ({current_length} chars)")
        return selected_items
    
    async def _generate_enhanced_response(self,
                                        query_text: str,
                                        context_items: List[ContextItem],
                                        mode: QueryMode,
                                        include_metadata: bool) -> Dict[str, Any]:
        """Generate enhanced response with detailed context analysis"""
        if not context_items:
            return {
                "status": "success",
                "response": "I couldn't find relevant information to answer your question.",
                "context_analysis": {
                    "items_used": 0,
                    "text_items": 0,
                    "image_items": 0,
                    "avg_relevance": 0.0
                }
            }
        
        # Prepare enhanced context
        context_text = self._format_enhanced_context(context_items)
        
        # Create sophisticated prompt
        prompt = self._create_enhanced_prompt(query_text, context_text, mode, context_items)
        
        # Generate response
        response = await self.unified_processor.llm_client.generate_response(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.7
        )
        
        # Analyze context usage
        context_analysis = self._analyze_context_usage(context_items)
        
        result = {
            "status": "success",
            "response": response,
            "query_mode": mode.value,
            "context_analysis": context_analysis
        }
        
        if include_metadata:
            result["context_items"] = [
                {
                    "content_type": item.content_type,
                    "source_file": item.source_file,
                    "relevance_score": round(item.similarity_score, 3),
                    "quality_score": round(item.quality_score, 3),
                    "content_preview": item.content[:100] + "..." if len(item.content) > 100 else item.content
                }
                for item in context_items
            ]
        
        return result
    
    def _format_enhanced_context(self, context_items: List[ContextItem]) -> str:
        """Format context items with enhanced structure"""
        context_parts = []
        
        for i, item in enumerate(context_items, 1):
            if item.content_type == "text":
                header = f"[Text Content {i}] (Relevance: {item.similarity_score:.2f})"
                context_parts.append(f"{header}\n{item.content}")
            elif item.content_type == "image":
                filename = item.metadata.get('image_filename', 'unknown')
                header = f"[Visual Element {i}] (File: {filename}, Relevance: {item.similarity_score:.2f})"
                context_parts.append(f"{header}\n{item.content}")
        
        return "\n\n".join(context_parts)
    
    def _create_enhanced_prompt(self, 
                              query_text: str, 
                              context_text: str, 
                              mode: QueryMode,
                              context_items: List[ContextItem]) -> str:
        """Create an enhanced prompt with sophisticated instructions"""
        
        text_count = sum(1 for item in context_items if item.content_type == 'text')
        image_count = sum(1 for item in context_items if item.content_type == 'image')
        
        mode_instructions = {
            QueryMode.HYBRID: f"The context includes both textual content ({text_count} items) and visual descriptions ({image_count} items). Integrate information from both sources to provide a comprehensive answer.",
            QueryMode.TEXT_ONLY: f"The context includes textual content ({text_count} items). Focus on the written information to answer the question.",
            QueryMode.IMAGE_ONLY: f"The context includes visual descriptions ({image_count} items). Focus on the visual elements and diagrams to answer the question.",
            QueryMode.AUTO: "The context has been automatically selected for optimal relevance."
        }
        
        instruction = mode_instructions.get(mode, "Use the provided context to answer the question.")
        
        prompt = f"""You are an advanced AI assistant that provides comprehensive answers based on document analysis. 

Context Information:
{context_text}

Query Mode: {mode.value}
Instructions: {instruction}

Question: {query_text}

Please provide a detailed, accurate answer that:
1. Directly addresses the question asked
2. Uses specific information from the provided context
3. Distinguishes between text-based and visual information when relevant
4. Maintains accuracy and avoids speculation beyond the provided context
5. Organizes the response clearly and logically

If the context doesn't contain sufficient information to answer the question, state this clearly and explain what information would be needed.

Response:"""
        
        return prompt
    
    def _analyze_context_usage(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Analyze context usage statistics"""
        if not context_items:
            return {
                "items_used": 0,
                "text_items": 0,
                "image_items": 0,
                "avg_relevance": 0.0,
                "total_chars": 0
            }
        
        text_items = [item for item in context_items if item.content_type == 'text']
        image_items = [item for item in context_items if item.content_type == 'image']
        
        avg_relevance = sum(item.similarity_score for item in context_items) / len(context_items)
        total_chars = sum(len(item.content) for item in context_items)
        
        return {
            "items_used": len(context_items),
            "text_items": len(text_items),
            "image_items": len(image_items),
            "avg_relevance": round(avg_relevance, 3),
            "total_chars": total_chars,
            "context_balance": {
                "text_ratio": len(text_items) / len(context_items),
                "image_ratio": len(image_items) / len(context_items)
            }
        }


# Test function
async def test_enhanced_query_engine():
    """Test the enhanced query engine"""
    from unified_pdf_processor import create_unified_pdf_processor
    
    # Create unified processor with mock mode
    processor = create_unified_pdf_processor(
        api_key=None,
        use_mock=True
    )
    
    # Create enhanced query engine
    query_engine = EnhancedMultimodalQueryEngine(processor)
    
    # Test with different query modes
    test_queries = [
        "What is machine learning?",
        "Show me the workflow diagrams",
        "Explain the neural network architecture with visual details"
    ]
    
    for query in test_queries:
        print(f"\nTesting advanced query: {query}")
        result = await query_engine.advanced_query(
            query,
            mode=QueryMode.AUTO,
            include_metadata=True
        )
        print(f"Status: {result.get('status')}")
        print(f"Mode used: {result.get('query_mode')}")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Context analysis: {result.get('context_analysis')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_enhanced_query_engine())