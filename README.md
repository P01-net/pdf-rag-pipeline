# PDF-Based Retrieval-Augmented Generation (RAG) System

A Python-based document question-answering system that uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDF documents and generate context-aware answers using Google's Gemini LLM.

## Overview

This project implements an end-to-end RAG pipeline that combines semantic search with Large Language Model (LLM) generation.

Instead of relying entirely on an LLM's pretrained knowledge, the application retrieves relevant document passages and supplies them as context to Gemini before generating an answer.

## Architecture

PDF Document → Text Extraction → Text Chunking → Embedding Generation → FAISS Vector Database → Similarity Search → Gemini LLM → Answer

## Tech Stack

- **Language:** Python
- **PDF Processing:** PyPDF
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **LLM:** Google Gemini API
- **Numerical Processing:** NumPy

## Features

- PDF text extraction with page-level tracking.
- Overlapping text chunking for context preservation.
- Dense vector embedding generation.
- FAISS-based semantic similarity search.
- Top-K document retrieval.
- Gemini-powered context-aware answer generation.
- Persistent storage of vector indexes and chunk metadata.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/document-rag.git
cd document-rag
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Usage

Place your PDF document in the project directory.

Run:

```bash
python rag_app.py sample.pdf
```

Ask questions interactively:

```text
Ask a question: What is the main topic of this document?

[RETRIEVED CONTEXT]
1. Page 2 | Similarity: 0.82
2. Page 4 | Similarity: 0.76
3. Page 1 | Similarity: 0.71

[GENERATED ANSWER]
Your context-grounded answer appears here.
```

To retrieve more context:

```bash
python rag_app.py sample.pdf --top-k 5
```

## Project Structure

```text
document-rag/
│
├── rag_app.py
├── requirements.txt
├── README.md
│
├── rag_index/
│   ├── faiss.index
│   └── chunks.json
│
└── sample.pdf
```

## Future Improvements

- Flask or FastAPI web interface.
- Support for multiple PDF documents.
- Conversational memory.
- Hybrid search using keyword and semantic retrieval.
- Reranking for improved retrieval relevance.
- Streaming LLM responses.
- Retrieval evaluation using Recall@K and MRR.

## Learning Outcomes

This project demonstrates practical implementation of document ingestion, text chunking, embedding generation, vector indexing, semantic retrieval, and LLM integration in a complete Retrieval-Augmented Generation workflow.