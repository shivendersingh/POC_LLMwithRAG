# 🤖 Advanced RAG System with DeepSeek API

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent document analysis and question-answering using **DeepSeek API**. Upload PDF documents and get AI-powered answers with proper citations and context.

![RAG System Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

### 🔥 Core Capabilities
- **📄 PDF Document Processing**: Extract text and analyze images from PDF files
- **🧠 Intelligent Q&A**: Ask questions about your documents in natural language
- **🎯 Context-Aware Responses**: Get accurate answers with source citations
- **⚡ High Performance**: Optimized with caching and efficient retrieval
- **🔒 Secure**: API keys protected, production-ready architecture

### 🚀 Advanced Features
- **Multi-Model Support**: DeepSeek (primary) + OpenAI (fallback)
- **Smart Document Chunking**: Intelligent text segmentation for better retrieval
- **Vector Similarity Search**: ChromaDB for semantic document search
- **Real-time Streaming**: Live response generation
- **Response Caching**: Faster repeated queries
- **Web Interface**: Beautiful Streamlit frontend

### 🎨 User Experience
- **Drag & Drop Upload**: Easy PDF document ingestion
- **Instant Processing**: Real-time document analysis
- **Interactive Chat**: Conversational interface
- **Performance Metrics**: Real-time system monitoring
- **Mobile Responsive**: Works on all devices

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG Core       │    │   DeepSeek API  │
│   Frontend      │────│   System         │────│   Integration   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼────┐ ┌──▼────────┐
            │    PDF    │ │Vector  │ │Generation │
            │ Processor │ │ Store  │ │   Layer   │
            └───────────┘ └────────┘ └───────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **DeepSeek API Key** (get one at [DeepSeek Platform](https://platform.deepseek.com))
- **Git** for cloning the repository

### 1️⃣ Clone Repository

```bash
git clone https://github.com/shivendersingh/POC_LLMwithRAG.git
cd POC_LLMwithRAG
```

### 2️⃣ Set Up Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configure API Keys

Copy `.env.example` to `.env` and add your DeepSeek API key:

```bash
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### 4️⃣ Run the System

```bash
streamlit run streamlit_app.py
```

## 📚 Documentation
- See `SECURITY.md` for security best practices
- See `DEPLOYMENT.md` for deployment instructions
- See `COMPLETION_SUMMARY.md` for project summary

## 🛡️ Security
- API keys are never committed to git
- Sensitive files are protected by `.gitignore`
- Follow `SECURITY.md` for compliance and best practices

## 📝 License
MIT

## 👤 Author
Shivender Singh (@shivendersingh)

---

**For questions or contributions, open an issue or pull request on GitHub!**
