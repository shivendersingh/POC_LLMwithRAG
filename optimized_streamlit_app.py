"""
Optimized Streamlit App for Multimodal RAG System

This is an optimized version of the Streamlit interface that uses the new
OptimizedRAGSystem with caching and duplicate prevention.
"""

import streamlit as st
import asyncio
import logging
import os
import tempfile
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Import optimized components
from optimized_rag_system import create_optimized_rag_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DeepSeek API configuration (using mock mode)
DEEPSEEK_API_KEY = None


@st.cache_resource
def initialize_rag_system():
    """Initialize the optimized RAG system (cached across sessions)"""
    try:
        logger.info("Initializing optimized RAG system...")
        rag_system = create_optimized_rag_system(
            deepseek_api_key=DEEPSEEK_API_KEY,
            cache_dir="./rag_cache"
        )
        logger.info("RAG system initialized successfully")
        return rag_system
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


def display_performance_stats(rag_system):
    """Display performance statistics in the sidebar"""
    if rag_system:
        stats = rag_system.get_processing_stats()
        
        st.sidebar.subheader("📊 Performance Stats")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Documents Processed", stats["documents_processed"])
            st.metric("Cache Hits", stats["cache_hits"])
        
        with col2:
            st.metric("Duplicates Prevented", stats["duplicate_prevents"])
            st.metric("Cache Hit Rate", stats["cache_hit_rate"])
        
        st.sidebar.metric("Total Cached", stats["total_cached_documents"])


def display_processing_progress(message: str, progress: Optional[float] = None):
    """Display processing progress with optional progress bar"""
    if progress is not None:
        st.progress(progress, text=message)
    else:
        with st.spinner(message):
            time.sleep(0.1)  # Brief pause for UI update


async def process_uploaded_file(uploaded_file, rag_system, force_reprocess: bool = False) -> Dict[str, Any]:
    """Process uploaded PDF file with optimization"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    try:
        # Check if document is already processed
        if not force_reprocess and rag_system.is_document_processed(temp_file_path):
            st.info("📋 Document already processed! Using cached results.")
            cached_result = rag_system.get_cached_document(temp_file_path)
            return {
                "status": "cached",
                "message": "Document was already processed",
                "cached_data": cached_result
            }
        
        # Process document with progress updates
        display_processing_progress("🔍 Analyzing document structure...", 0.2)
        
        display_processing_progress("📄 Processing text content...", 0.4)
        
        display_processing_progress("🖼️ Processing image content...", 0.6)
        
        display_processing_progress("💾 Creating embeddings and storing...", 0.8)
        
        # Actual processing
        result = await rag_system.process_document_optimized(
            temp_file_path,
            chunk_size=1000,
            chunk_overlap=200,
            force_reprocess=force_reprocess
        )
        
        display_processing_progress("✅ Processing complete!", 1.0)
        
        return result
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")


async def query_documents(question: str, rag_system, query_mode: str = "auto", enable_vqa: bool = True) -> Dict[str, Any]:
    """Query documents with VQA support and mode selection"""
    
    display_processing_progress("🔍 Searching relevant content...")
    
    # Pass the query mode and VQA settings to the processor
    result = await rag_system.query_documents(
        question=question,
        max_results=5,
        similarity_threshold=0.7,
        query_mode=query_mode,
        enable_vqa=enable_vqa
    )
    
    return result


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Optimized Multimodal RAG System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("🚀 Optimized Multimodal RAG System")
    st.markdown("*Enhanced with caching, duplicate prevention, and performance optimizations*")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("❌ Failed to initialize RAG system. Please check the logs.")
        return
    
    # Sidebar with performance stats and controls
    st.sidebar.title("🎛️ System Controls")
    
    # Performance statistics
    display_performance_stats(rag_system)
    
    # Query mode settings
    st.sidebar.subheader("🎯 Query Settings")
    
    query_mode = st.sidebar.selectbox(
        "Query Mode",
        options=["auto", "hybrid", "text_only", "image_only"],
        index=0,
        help="Select how to process queries:\n- Auto: Smart detection\n- Hybrid: Text + Images\n- Text Only: Text content only\n- Image Only: Images only"
    )
    
    enable_vqa = st.sidebar.checkbox(
        "🔍 Enable Visual Q&A",
        value=True,
        help="Enable direct visual question answering for image-specific queries"
    )
    
    if enable_vqa:
        st.sidebar.info("💡 VQA Mode: Ask specific questions about images like 'What color is the sun?' or 'What planets are shown?'")
    
    # Cache management
    st.sidebar.subheader("🗂️ Cache Management")
    
    if st.sidebar.button("🔄 Refresh Stats"):
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Cache", type="secondary"):
        if st.sidebar.checkbox("⚠️ Confirm cache clear"):
            if rag_system.clear_cache(confirm=True):
                st.sidebar.success("Cache cleared successfully!")
                st.rerun()
            else:
                st.sidebar.error("Failed to clear cache")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to process with the multimodal RAG system"
        )
        
        # Processing options
        force_reprocess = st.checkbox(
            "🔄 Force reprocess", 
            value=False,
            help="Force reprocessing even if document is already cached"
        )
        
        if uploaded_file is not None:
            st.success(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if st.button("🚀 Process Document", type="primary"):
                with st.container():
                    start_time = time.time()
                    
                    # Process the document
                    result = asyncio.run(
                        process_uploaded_file(uploaded_file, rag_system, force_reprocess)
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Display results
                    if result["status"] == "success":
                        st.success("✅ Document processed successfully!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Text Chunks", result.get("chunks_processed", 0))
                        with col_b:
                            st.metric("Images Processed", result.get("images_processed", 0))
                        with col_c:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                        # Store in session state for querying
                        st.session_state["document_processed"] = True
                        st.session_state["last_processed"] = uploaded_file.name
                        
                    elif result["status"] == "cached":
                        st.info("📋 Using cached results!")
                        cached_data = result.get("cached_data", {})
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Text Chunks", cached_data.get("chunks_processed", "N/A"))
                        with col_b:
                            st.metric("Images Processed", cached_data.get("images_processed", "N/A"))
                        with col_c:
                            st.metric("Cache Hit", "✅")
                        
                        st.session_state["document_processed"] = True
                        st.session_state["last_processed"] = uploaded_file.name
                        
                    else:
                        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.header("❓ Question & Answer")
        
        # Check if any document is processed
        if st.session_state.get("document_processed", False):
            st.success(f"✅ Ready to answer questions about: {st.session_state.get('last_processed', 'processed document')}")
            
            question = st.text_area(
                "Ask a question about your document:",
                placeholder="What is this document about?\nSummarize the main points.\nWhat charts or diagrams are included?",
                height=100
            )
            
            if question and st.button("🔍 Get Answer", type="primary"):
                start_time = time.time()
                
                with st.spinner("Processing your question... This may take a moment for visual content."):
                    # Show query mode indicator
                    query_mode_text = f"Query Mode: {query_mode.title()}"
                    if enable_vqa:
                        query_mode_text += " | VQA Enabled 🔍"
                    st.info(query_mode_text)
                    
                    # Query the documents with configured settings
                    result = asyncio.run(query_documents(
                        question, 
                        rag_system,
                        query_mode=query_mode,
                        enable_vqa=enable_vqa
                    ))
                
                query_time = time.time() - start_time
                
                # Display results
                if result["status"] == "success":
                    st.success("✅ Answer generated successfully!")
                    
                    # Enhanced metrics display
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Query Time", f"{query_time:.2f}s")
                    with col_b:
                        st.metric("Context Chunks", result.get("context_chunks", 0))
                    with col_c:
                        if result.get("used_vqa", False):
                            st.metric("VQA Used", "✅")
                        else:
                            st.metric("Mode", result.get("query_mode", "Auto").title())
                    
                    # Display answer
                    st.subheader("💡 Answer:")
                    st.markdown(result["answer"])
                    
                    # Show additional VQA information if used
                    if result.get("used_vqa", False):
                        st.info("🔍 This answer was enhanced using Visual Question Answering on document images.")
                    
                    # Show document type information
                    if result.get("document_type"):
                        st.caption(f"Document Type: {result['document_type'].replace('_', ' ').title()}")
                    
                    # Display context (optional)
                    if st.expander("📚 View Source Context"):
                        for i, source in enumerate(result.get("search_results", []), 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{source['text'][:300]}...\n```")
                            st.markdown(f"*Distance: {source.get('distance', 'N/A'):.3f}*")
                            st.divider()
                
                elif result["status"] == "no_results":
                    st.warning("⚠️ No relevant content found in the processed documents.")
                    
                else:
                    st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        else:
            st.info("📋 Please upload and process a PDF document first to enable Q&A.")
    
    # Footer with system information
    st.divider()
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns(3)
    
    with col_foot1:
        st.markdown("**🔧 System Status:**")
        st.markdown("✅ Optimized RAG System Active")
        
    with col_foot2:
        st.markdown("**⚡ Performance Features:**")
        st.markdown("• Document caching")
        st.markdown("• Duplicate prevention")
        st.markdown("• Optimized image processing")
        
    with col_foot3:
        st.markdown("**🔗 Components:**")
        st.markdown("• DeepSeek LLM + Vision")
        st.markdown("• ChromaDB Vector Store")
        st.markdown("• Visual Question Answering")
        st.markdown("• OCR Text Extraction")


if __name__ == "__main__":
    main()