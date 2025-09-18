"""
Task D: Simple UI for Document Upload and Q&A
Streamlit-based web interface for the unified multimodal RAG system.

This interface provides:
1. PDF document upload capability
2. Document processing with text and image extraction
3. Interactive Q&A with multiple query modes
4. Real-time processing status and results display
"""

import streamlit as st
import os
import tempfile
import time
from typing import Optional, Dict, Any
import asyncio
from pathlib import Path

# Import our unified multimodal processor from Task C
from unified_pdf_processor import create_unified_pdf_processor, UnifiedPDFProcessor

class StreamlitRAGApp:
    """Streamlit application for the unified multimodal RAG system."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="streamlit_rag_")
        
    def initialize_processor(self) -> UnifiedPDFProcessor:
        """Initialize the unified PDF processor."""
        if 'processor' not in st.session_state:
            with st.spinner("🔧 Initializing multimodal RAG system..."):
                st.session_state.processor = create_unified_pdf_processor(
                    api_key=self.api_key,
                    use_mock=False,
                    collection_name="streamlit_app_collection"
                )
        return st.session_state.processor
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="📚 Multimodal PDF Q&A System",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def render_header(self):
        """Render the application header."""
        st.title("📚 Multimodal PDF Q&A System")
        st.markdown("""
        **Welcome to the Task D Interface!** 
        
        This system combines:
        - 📝 **Task A**: PDF Text Processing
        - 🖼️ **Task B**: PDF Image Processing  
        - 🔄 **Task C**: Unified Multimodal Processing
        - 🌐 **Task D**: Interactive Web Interface
        """)
        
    def render_sidebar(self):
        """Render the sidebar with system information."""
        with st.sidebar:
            st.header("🔧 System Status")
            
            # Check if processor is initialized
            processor_status = "✅ Ready" if 'processor' in st.session_state else "⏳ Not Initialized"
            st.write(f"**Processor**: {processor_status}")
            
            # Show document status
            doc_status = "✅ Loaded" if 'processed_doc' in st.session_state else "📄 No Document"
            st.write(f"**Document**: {doc_status}")
            
            if 'processed_doc' in st.session_state:
                result = st.session_state.processed_doc
                st.write(f"**Text Chunks**: {result.text_chunks}")
                st.write(f"**Images**: {result.image_count}")
                st.write(f"**Processing Time**: {result.processing_time:.2f}s")
            
            st.divider()
            
            # Query modes explanation
            st.header("🔍 Query Modes")
            st.write("**Text-Only** 📝: Search text content only")
            st.write("**Image-Only** 🖼️: Search image descriptions only")  
            st.write("**Hybrid** 🔄: Search both text and images")
            st.write("**Auto** 🤖: Smart mode detection")
            
            st.divider()
            
            # System capabilities
            st.header("⚡ Capabilities")
            st.write("✅ PDF Text Extraction")
            st.write("✅ Image Extraction & Analysis")
            st.write("✅ Multimodal Vector Search")
            st.write("✅ DeepSeek LLM Integration")
            st.write("✅ Real-time Q&A")
            
    def render_upload_section(self) -> Optional[str]:
        """Render the PDF upload section."""
        st.header("📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze. The system will extract both text and images."
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            temp_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            return temp_path
            
        return None
    
    async def process_document(self, file_path: str, processor: UnifiedPDFProcessor):
        """Process the uploaded document."""
        st.header("⚙️ Document Processing")
        
        # Create processing status containers
        status_container = st.container()
        progress_bar = st.progress(0)
        
        with status_container:
            st.info("🔄 Starting multimodal document processing...")
        
        try:
            # Start processing
            progress_bar.progress(10)
            
            with status_container:
                st.info("📝 Extracting text content...")
            
            # Process the document
            result = await processor.process_pdf_complete(file_path)
            progress_bar.progress(100)
            
            # Store result in session state
            st.session_state.processed_doc = result
            
            if result.status == "success":
                with status_container:
                    st.success("✅ Document processed successfully!")
                
                # Display processing results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Text Chunks", result.text_chunks)
                with col2:
                    st.metric("🖼️ Images Found", result.image_count)
                with col3:
                    st.metric("⏱️ Processing Time", f"{result.processing_time:.2f}s")
                
                if result.errors:
                    st.warning(f"⚠️ Some issues occurred: {result.errors}")
                    
                return True
            else:
                with status_container:
                    st.error("❌ Document processing failed!")
                if result.errors:
                    st.error(f"Errors: {result.errors}")
                return False
                
        except Exception as e:
            progress_bar.progress(0)
            with status_container:
                st.error(f"❌ Processing failed: {str(e)}")
            return False
    
    def render_query_section(self, processor: UnifiedPDFProcessor):
        """Render the Q&A section."""
        st.header("❓ Ask Questions")
        
        if 'processed_doc' not in st.session_state:
            st.warning("⚠️ Please upload and process a document first.")
            return
        
        # Query mode selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                help="Ask any question about the uploaded document"
            )
        
        with col2:
            query_mode = st.selectbox(
                "Query Mode:",
                ["auto", "hybrid", "text_only", "image_only"],
                help="Choose how to search the document"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Results to retrieve", 1, 10, 5)
            with col2:
                temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)
        
        # Query button
        if st.button("🔍 Ask Question", type="primary", disabled=not question):
            self.handle_query(question, query_mode, top_k, processor)
    
    async def handle_query(self, question: str, mode: str, top_k: int, processor: UnifiedPDFProcessor):
        """Handle the user query and display results."""
        
        # Create response containers
        response_container = st.container()
        
        with response_container:
            with st.spinner(f"🔍 Searching in {mode} mode..."):
                try:
                    # Execute query
                    start_time = time.time()
                    response = await processor.query(
                        question, 
                        mode=mode, 
                        top_k=top_k
                    )
                    end_time = time.time()
                    
                    # Display results
                    if isinstance(response, dict) and 'response' in response:
                        st.success("✅ Query completed successfully!")
                        
                        # Response
                        st.subheader("💬 Answer")
                        st.markdown(response['response'])
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⏱️ Response Time", f"{end_time - start_time:.2f}s")
                        with col2:
                            if 'text_sources' in response:
                                st.metric("📝 Text Sources", response['text_sources'])
                        with col3:
                            if 'image_sources' in response:
                                st.metric("🖼️ Image Sources", response['image_sources'])
                        
                        # Store in session state for history
                        if 'query_history' not in st.session_state:
                            st.session_state.query_history = []
                        
                        st.session_state.query_history.append({
                            'question': question,
                            'mode': mode,
                            'response': response['response'],
                            'timestamp': time.time()
                        })
                        
                    else:
                        st.warning("⚠️ Received unexpected response format")
                        st.json(response)
                        
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
    
    def render_history_section(self):
        """Render the query history section."""
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.header("📋 Query History")
            
            for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                with st.expander(f"Q{len(st.session_state.query_history)-i}: {entry['question'][:50]}..."):
                    st.write(f"**Mode**: {entry['mode']}")
                    st.write(f"**Question**: {entry['question']}")
                    st.write(f"**Answer**: {entry['response']}")
                    st.write(f"**Time**: {time.ctime(entry['timestamp'])}")
    
    async def run(self):
        """Main application runner."""
        # Setup page
        self.setup_page_config()
        self.render_header() 
        self.render_sidebar()
        
        # Initialize processor
        processor = self.initialize_processor()
        
        # Document upload
        uploaded_file_path = self.render_upload_section()
        
        # Process document if uploaded
        if uploaded_file_path and ('processed_doc' not in st.session_state or 
                                   st.session_state.get('last_processed_file') != uploaded_file_path):
            
            if await self.process_document(uploaded_file_path, processor):
                st.session_state.last_processed_file = uploaded_file_path
                st.rerun()  # Refresh to show processed state
        
        # Query section
        self.render_query_section(processor)
        
        # History section
        self.render_history_section()
        
        # Footer
        st.divider()
        st.markdown("""
        **🎉 Congratulations!** You've successfully implemented all tasks:
        - ✅ **Task A**: PDF Text Processing
        - ✅ **Task B**: PDF Image Processing  
        - ✅ **Task C**: Unified Multimodal Processing
        - ✅ **Task D**: Interactive Web Interface
        
        The complete multimodal RAG system is now operational! 🚀
        """)

# Async wrapper for Streamlit
def run_async_app():
    """Run the async Streamlit app."""
    app = StreamlitRAGApp()
    
    # Create event loop for async operations
    if 'event_loop' not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)
    
    # Run the app synchronously (Streamlit doesn't support async directly)
    try:
        # We need to handle this differently since Streamlit doesn't support async main functions
        # Let's create a sync version
        app_sync = StreamlitRAGAppSync()
        app_sync.run()
    except Exception as e:
        st.error(f"Application error: {e}")

class StreamlitRAGAppSync:
    """Synchronous version of the Streamlit app."""
    
    def __init__(self):
    self.api_key = "sk-73419d2868334e3a93e1d4eb9bf6e4da"
        self.temp_dir = tempfile.mkdtemp(prefix="streamlit_rag_")
        
    def initialize_processor(self) -> UnifiedPDFProcessor:
        """Initialize the unified PDF processor."""
        if 'processor' not in st.session_state:
            with st.spinner("🔧 Initializing multimodal RAG system..."):
                st.session_state.processor = create_unified_pdf_processor(
                    api_key=self.api_key,
                    use_mock=False,
                    collection_name="streamlit_app_collection"
                )
        return st.session_state.processor
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="📚 Multimodal PDF Q&A System",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def render_header(self):
        """Render the application header."""
        st.title("📚 Multimodal PDF Q&A System")
        st.markdown("""
        **Welcome to the Task D Interface!** 
        
        This system combines:
        - 📝 **Task A**: PDF Text Processing
        - 🖼️ **Task B**: PDF Image Processing  
        - 🔄 **Task C**: Unified Multimodal Processing
        - 🌐 **Task D**: Interactive Web Interface
        """)
        
    def render_sidebar(self):
        """Render the sidebar with system information."""
        with st.sidebar:
            st.header("🔧 System Status")
            
            # Check if processor is initialized
            processor_status = "✅ Ready" if 'processor' in st.session_state else "⏳ Not Initialized"
            st.write(f"**Processor**: {processor_status}")
            
            # Show document status
            doc_status = "✅ Loaded" if 'processed_doc' in st.session_state else "📄 No Document"
            st.write(f"**Document**: {doc_status}")
            
            if 'processed_doc' in st.session_state:
                result = st.session_state.processed_doc
                st.write(f"**Text Chunks**: {result.text_chunks}")
                st.write(f"**Images**: {result.image_count}")
                st.write(f"**Processing Time**: {result.processing_time:.2f}s")
            
            st.divider()
            
            # Query modes explanation
            st.header("🔍 Query Modes")
            st.write("**Text-Only** 📝: Search text content only")
            st.write("**Image-Only** 🖼️: Search image descriptions only")  
            st.write("**Hybrid** 🔄: Search both text and images")
            st.write("**Auto** 🤖: Smart mode detection")
            
            st.divider()
            
            # System capabilities
            st.header("⚡ Capabilities")
            st.write("✅ PDF Text Extraction")
            st.write("✅ Image Extraction & Analysis")
            st.write("✅ Multimodal Vector Search")
            st.write("✅ DeepSeek LLM Integration")
            st.write("✅ Real-time Q&A")
            
    def render_upload_section(self) -> Optional[str]:
        """Render the PDF upload section."""
        st.header("📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze. The system will extract both text and images."
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            temp_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            return temp_path
            
        return None
    
    def process_document(self, file_path: str, processor: UnifiedPDFProcessor):
        """Process the uploaded document (sync version)."""
        st.header("⚙️ Document Processing")
        
        # Create processing status containers
        status_container = st.container()
        progress_bar = st.progress(0)
        
        with status_container:
            st.info("🔄 Starting multimodal document processing...")
        
        try:
            # Start processing
            progress_bar.progress(10)
            
            with status_container:
                st.info("📝 Processing document (this may take a moment)...")
            
            # Process the document using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(processor.process_pdf_complete(file_path))
            loop.close()
            
            progress_bar.progress(100)
            
            # Store result in session state
            st.session_state.processed_doc = result
            
            if result.status == "success":
                with status_container:
                    st.success("✅ Document processed successfully!")
                
                # Display processing results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Text Chunks", result.text_chunks)
                with col2:
                    st.metric("🖼️ Images Found", result.image_count)
                with col3:
                    st.metric("⏱️ Processing Time", f"{result.processing_time:.2f}s")
                
                if result.errors:
                    st.warning(f"⚠️ Some issues occurred: {result.errors}")
                    
                return True
            else:
                with status_container:
                    st.error("❌ Document processing failed!")
                if result.errors:
                    st.error(f"Errors: {result.errors}")
                return False
                
        except Exception as e:
            progress_bar.progress(0)
            with status_container:
                st.error(f"❌ Processing failed: {str(e)}")
            return False
    
    def render_query_section(self, processor: UnifiedPDFProcessor):
        """Render the Q&A section."""
        st.header("❓ Ask Questions")
        
        if 'processed_doc' not in st.session_state:
            st.warning("⚠️ Please upload and process a document first.")
            return
        
        # Query mode selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                help="Ask any question about the uploaded document"
            )
        
        with col2:
            query_mode = st.selectbox(
                "Query Mode:",
                ["auto", "hybrid", "text_only", "image_only"],
                help="Choose how to search the document"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Results to retrieve", 1, 10, 5)
            with col2:
                temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)
        
        # Query button
        if st.button("🔍 Ask Question", type="primary", disabled=not question):
            self.handle_query(question, query_mode, top_k, processor)
    
    def handle_query(self, question: str, mode: str, top_k: int, processor: UnifiedPDFProcessor):
        """Handle the user query and display results."""
        
        # Create response containers
        response_container = st.container()
        
        with response_container:
            with st.spinner(f"🔍 Searching in {mode} mode..."):
                try:
                    # Execute query
                    start_time = time.time()
                    
                    # Run async query in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(processor.query(
                        question, 
                        mode=mode, 
                        top_k=top_k
                    ))
                    loop.close()
                    
                    end_time = time.time()
                    
                    # Display results
                    if isinstance(response, dict) and 'response' in response:
                        st.success("✅ Query completed successfully!")
                        
                        # Response
                        st.subheader("💬 Answer")
                        st.markdown(response['response'])
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⏱️ Response Time", f"{end_time - start_time:.2f}s")
                        with col2:
                            if 'text_sources' in response:
                                st.metric("📝 Text Sources", response['text_sources'])
                        with col3:
                            if 'image_sources' in response:
                                st.metric("🖼️ Image Sources", response['image_sources'])
                        
                        # Store in session state for history
                        if 'query_history' not in st.session_state:
                            st.session_state.query_history = []
                        
                        st.session_state.query_history.append({
                            'question': question,
                            'mode': mode,
                            'response': response['response'],
                            'timestamp': time.time()
                        })
                        
                    else:
                        st.warning("⚠️ Received unexpected response format")
                        st.json(response)
                        
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
    
    def render_history_section(self):
        """Render the query history section."""
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.header("📋 Query History")
            
            for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                with st.expander(f"Q{len(st.session_state.query_history)-i}: {entry['question'][:50]}..."):
                    st.write(f"**Mode**: {entry['mode']}")
                    st.write(f"**Question**: {entry['question']}")
                    st.write(f"**Answer**: {entry['response']}")
                    st.write(f"**Time**: {time.ctime(entry['timestamp'])}")
    
    def run(self):
        """Main application runner."""
        # Setup page
        self.setup_page_config()
        self.render_header() 
        self.render_sidebar()
        
        # Initialize processor
        processor = self.initialize_processor()
        
        # Document upload
        uploaded_file_path = self.render_upload_section()
        
        # Process document if uploaded
        if uploaded_file_path and ('processed_doc' not in st.session_state or 
                                   st.session_state.get('last_processed_file') != uploaded_file_path):
            
            if self.process_document(uploaded_file_path, processor):
                st.session_state.last_processed_file = uploaded_file_path
                st.rerun()  # Refresh to show processed state
        
        # Query section
        self.render_query_section(processor)
        
        # History section
        self.render_history_section()
        
        # Footer
        st.divider()
        st.markdown("""
        **🎉 Congratulations!** You've successfully implemented all tasks:
        - ✅ **Task A**: PDF Text Processing
        - ✅ **Task B**: PDF Image Processing  
        - ✅ **Task C**: Unified Multimodal Processing
        - ✅ **Task D**: Interactive Web Interface
        
        The complete multimodal RAG system is now operational! 🚀
        """)

# Main entry point
if __name__ == "__main__":
    app = StreamlitRAGAppSync()
    app.run()