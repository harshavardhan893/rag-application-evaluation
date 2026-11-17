@'
# RAG Application Evaluation

## Project Overview
This project implements and evaluates a Retrieval-Augmented Generation (RAG) application using fictional company policy documents.

## Technology Stack
- Python 3.12
- LangChain
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS Vector Database
- Ollama (Qwen2.5:3B)
- Streamlit
- Pandas and Matplotlib

## Architecture
1. Load company policy documents.
2. Split documents into chunks (500 characters, 100 overlap).
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in FAISS.
5. Retrieve the top 3 relevant chunks.
6. Generate answers using Qwen2.5:3B.
7. Display answers and sources through Streamlit.

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/harshavardhan893/rag-application-evaluation.git
cd rag-application-evaluation