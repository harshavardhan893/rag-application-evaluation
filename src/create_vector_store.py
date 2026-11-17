
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from chunk_documents import chunk_documents


PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"


def create_vector_store():
    chunks = chunk_documents()

    texts = [chunk["content"] for chunk in chunks]

    metadata = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadata
    )

    vector_store.save_local(str(VECTOR_STORE_DIR))

    print(f"Total chunks stored: {len(texts)}")
    print(f"Vector store saved at: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    create_vector_store()
