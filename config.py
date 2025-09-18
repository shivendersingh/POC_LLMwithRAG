"""
Configuration file for the RAG system with DeepSeek API
"""


# PDF Path Configuration
PDF_PATH = r"E:\POC_LLMwithRAG\DemoData\knowledge.pdf"

# System Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_COLLECTION = "deepseek_rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Logging Configuration
LOG_LEVEL = "INFO"