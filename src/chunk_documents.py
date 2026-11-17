
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import load_documents


def chunk_documents():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_{index}",
                "content": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_documents()

    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:
        print(f"\nSource: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Content: {chunk['content'][:150]}...")
