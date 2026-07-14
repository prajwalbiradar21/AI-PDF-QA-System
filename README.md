# 📄 AI PDF Question Answering System

An AI-powered document intelligence application that enables users to upload PDF documents and ask natural language questions based on the document content. The system uses **Retrieval-Augmented Generation (RAG)** with **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Llama 3.1 (Groq)** to generate accurate, context-aware responses.

---

##  Project Overview

Searching through lengthy PDF documents manually is time-consuming and inefficient. This project simplifies document analysis by allowing users to interact with PDF files using natural language.

The application extracts text from uploaded PDFs, converts the content into vector embeddings, retrieves the most relevant information using semantic search, and generates contextual answers using a Large Language Model (LLM).

---

##  Features

-  📄 Upload PDF documents
-  💬 Ask questions in natural language
-  🔍 Semantic search using vector embeddings
-  🧠 Retrieval-Augmented Generation (RAG)
-  ⚡ Fast similarity search using FAISS
-  🤖  Context-aware AI responses using Llama 3.1
-  📚 Source references for generated answers
-  🌐 Simple and interactive Streamlit interface

---

##  Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| Streamlit | User Interface |
| LangChain | LLM Orchestration |
| PyPDF | PDF Text Extraction |
| HuggingFace Embeddings | Text Embedding Generation |
| FAISS | Vector Database |
| Groq API | Llama 3.1 Inference |
| NLP | Semantic Search |

---

##  System Workflow

```
User Uploads PDF
        │
        ▼
Extract Text using PyPDFLoader
        │
        ▼
Split Text into Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Store Embeddings in FAISS
        │
        ▼
User Asks Question
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
Generate Answer using Llama 3.1
        │
        ▼
Display Answer with Source References
```

---

## 📂 Project Structure

```text
AI-PDF-QA-System/
│
├── screenshots/
│   ├── home-page.png
│   ├── pdf-upload.png
│   └── ai-response.png
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```