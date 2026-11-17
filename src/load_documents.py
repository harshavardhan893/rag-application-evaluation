
from pathlib import Path

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Folder containing knowledge-base documents
DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "content": text
        })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Total documents loaded: {len(docs)}")

    for doc in docs:
        print(f"\nDocument: {doc['source']}")
        print(f"Content length: {len(doc['content'])} characters")
