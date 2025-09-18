"""
PDF Text Processing Module for RAG System

This module implements SimplePDFTextProcessor for extracting, chunking, embedding,
and storing text from PDF documents for RAG applications.
"""

import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract base class for vector storage systems"""
    
    @abstractmethod
    def add_embedding(self, embedding: List[float], text: str, metadata: Dict[str, Any]) -> str:
        """Add an embedding to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger = logging.getLogger(__name__)
    
    def add_embedding(self, embedding: List[float], text: str, metadata: Dict[str, Any]) -> str:
        """Add an embedding to ChromaDB"""
        chunk_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[chunk_id]
        )
        
        self.logger.info(f"Added embedding for chunk {chunk_id}")
        return chunk_id
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None
                })
        
        return formatted_results


class EmbeddingFunction:
    """Wrapper for sentence transformer embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Run embedding generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self.model.encode, 
            texts
        )
        
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]


class SimplePDFTextProcessor:
    """
    Core PDF text processing class for RAG applications.
    
    Handles PDF text extraction, chunking, embedding generation,
    and vector storage for retrieval-augmented generation.
    """
    
    def __init__(self, embedding_func: EmbeddingFunction, vector_store: VectorStore):
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    async def process_pdf_text(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """
        Extract and process text from PDF
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks processed
        """
        self.logger.info(f"Starting PDF processing for: {pdf_path}")
        
        # 1. Extract text from PDF
        text_content = await self._extract_text_from_pdf(pdf_path)
        
        if not text_content.strip():
            self.logger.warning(f"No text content extracted from {pdf_path}")
            return 0
        
        # 2. Split text into chunks
        chunks = self._split_text(text_content, chunk_size, chunk_overlap)
        self.logger.info(f"Created {len(chunks)} chunks from PDF")
        
        # 3. Generate embeddings and store
        doc_id = await self._embed_and_store(chunks, pdf_path)
        
        self.logger.info(f"Successfully processed PDF {pdf_path} with document ID: {doc_id}")
        return len(chunks)
    
    async def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF using PyPDF2"""
        self.logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.logger.info(f"PDF has {len(reader.pages)} pages")
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content.append(page_text)
                        self.logger.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            
            full_text = "\n".join(text_content)
            self.logger.info(f"Total text extracted: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine the end position for this chunk
            end = min(start + chunk_size, len(text))
            
            # Extract the chunk
            chunk = text[start:end]
            
            # Try to break at sentence boundaries if possible
            if end < len(text) and not text[end].isspace():
                # Look for the last sentence ending within the chunk
                last_sentence_end = max(
                    chunk.rfind('.'),
                    chunk.rfind('!'),
                    chunk.rfind('?')
                )
                
                if last_sentence_end > len(chunk) * 0.5:  # Only break if we're not losing too much
                    chunk = chunk[:last_sentence_end + 1]
                    end = start + last_sentence_end + 1
            
            chunks.append(chunk.strip())
            
            # Move start position considering overlap
            if end >= len(text):
                break
            start = end - chunk_overlap
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        self.logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    async def _embed_and_store(self, chunks: List[str], source_file: str) -> str:
        """
        Generate embeddings and store in vector database
        
        Args:
            chunks: List of text chunks
            source_file: Path to source file
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex}"
        self.logger.info(f"Embedding and storing {len(chunks)} chunks with document ID: {doc_id}")
        
        try:
            # Generate embeddings for all chunks
            embeddings = await self.embedding_func(chunks)
            
            # Store in vector database with metadata
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "source": source_file,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "content_type": "text",
                    "chunk_length": len(chunk),
                    "created_at": str(uuid.uuid1().time)
                }
                
                self.vector_store.add_embedding(
                    embedding=embedding,
                    text=chunk,
                    metadata=metadata
                )
            
            self.logger.info(f"Successfully stored {len(chunks)} chunks for document {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error embedding and storing chunks: {str(e)}")
            raise


class PDFQueryEngine:
    """Query engine for retrieving relevant text chunks from processed PDFs"""
    
    def __init__(self, embedding_func: EmbeddingFunction, vector_store: VectorStore):
        self.embedding_func = embedding_func
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    async def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant text chunks
        
        Args:
            query_text: Query string
            k: Number of results to return
            
        Returns:
            List of relevant text chunks with metadata
        """
        self.logger.info(f"Querying for: {query_text}")
        
        # Generate embedding for query
        query_embeddings = await self.embedding_func([query_text])
        query_embedding = query_embeddings[0]
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        self.logger.info(f"Found {len(results)} relevant chunks")
        return results


# Factory function to create a complete PDF processing system
def create_pdf_processor(model_name: str = "all-MiniLM-L6-v2", 
                        collection_name: str = "pdf_documents") -> tuple[SimplePDFTextProcessor, PDFQueryEngine]:
    """
    Factory function to create a complete PDF processing system
    
    Args:
        model_name: Name of the sentence transformer model
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Tuple of (processor, query_engine)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create components
    embedding_func = EmbeddingFunction(model_name)
    vector_store = ChromaVectorStore(collection_name)
    
    # Create processor and query engine
    processor = SimplePDFTextProcessor(embedding_func, vector_store)
    query_engine = PDFQueryEngine(embedding_func, vector_store)
    
    return processor, query_engine


if __name__ == "__main__":
    import asyncio
    
    async def test_pdf_processor():
        """Test the PDF processor with a sample file"""
        # Create processor and query engine
        processor, query_engine = create_pdf_processor()
        
        # Process a PDF (you'll need to provide a valid PDF path)
        pdf_path = "sample.pdf"  # Replace with actual PDF path
        
        try:
            chunk_count = await processor.process_pdf_text(pdf_path)
            print(f"Processed {chunk_count} chunks from {pdf_path}")
            
            # Test querying
            results = await query_engine.query("what is this document about?", k=3)
            
            print("\nQuery results:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Text: {result['text'][:200]}...")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Distance: {result.get('distance', 'N/A')}")
                
        except FileNotFoundError:
            print(f"PDF file not found: {pdf_path}")
            print("Please provide a valid PDF file path to test the processor")
    
    # Run the test
    asyncio.run(test_pdf_processor())