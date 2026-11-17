
from pathlib import Path
import time
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_DIR / "data" / "evaluation" / "evaluation_dataset.csv"
)

VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"
RESULTS_DIR = PROJECT_DIR / "results"


def evaluate_answers():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATASET_PATH)

    print("Total evaluation questions:", len(dataset))

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Loading FAISS vector database...")

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    print("Initializing local Qwen model...")

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a company IT support assistant.

Answer the question using ONLY the supplied company policy context.
Do not invent information or use outside knowledge.

If the answer is not available in the context, say:
"I don't have enough information in the company policies."

Context:
{context}

Question:
{question}

Answer:
""")

    results = []

    for index, row in dataset.iterrows():

        question = str(row["question"])
        reference_answer = str(row["reference_answer"])
        expected_source = str(row["source_document"])

        print(
            f"\nProcessing question {index + 1}/{len(dataset)}:"
        )
        print(question)

        start_time = time.time()

        documents = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        response = llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        )

        answer = response.content

        elapsed_time = time.time() - start_time

        retrieved_sources = list(dict.fromkeys(
            doc.metadata.get("source", "Unknown")
            for doc in documents
        ))

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": answer,
            "expected_source": expected_source,
            "retrieved_sources": " | ".join(retrieved_sources),
            "retrieved_context": context,
            "response_time_seconds": round(elapsed_time, 2)
        })

        print("Generated answer:", answer)
        print("Response time:", round(elapsed_time, 2), "seconds")

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "answer_evaluation_results.csv"

    results_df.to_csv(output_path, index=False)

    print("\n========================================")
    print("ANSWER GENERATION COMPLETED")
    print("========================================")

    print("Questions processed:", len(results_df))
    print("Results saved to:", output_path)

    print("\nPreview:")
    print(
        results_df[
            ["question", "generated_answer", "response_time_seconds"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    evaluate_answers()
