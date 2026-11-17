from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"


def test_retrieval():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    question = "How many incorrect password attempts cause an account lockout?"

    results = vector_store.similarity_search(question, k=3)

    print("\nQuestion:", question)
    print("\nRetrieved documents:")

    for result in results:
        print("\nSource:", result.metadata["source"])
        print("Content:", result.page_content)


if __name__ == "__main__":
    test_retrieval()
    