
from pathlib import Path
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"
DATASET_PATH = PROJECT_DIR / "data" / "evaluation" / "evaluation_dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"

K_VALUES = [1, 3, 5]


def evaluate_retrieval():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load evaluation dataset
    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {"question", "reference_answer", "source_document"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    print(f"Evaluation questions: {len(dataset)}")

    # Load embedding model
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load FAISS vector database
    print("Loading FAISS vector database...")
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    max_k = max(K_VALUES)
    detailed_results = []
    summary_results = []

    # Evaluate every question
    for index, row in dataset.iterrows():

        question = str(row["question"])
        expected_source = str(row["source_document"]).strip()

        print(f"\nQuestion {index + 1}/{len(dataset)}: {question}")

        retrieved_docs = vector_store.similarity_search(
            question,
            k=max_k
        )

        retrieved_sources = [
            doc.metadata.get("source", "")
            for doc in retrieved_docs
        ]

        # Rank of the first retrieved chunk from the expected source
        relevant_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if source == expected_source
        ]

        first_relevant_rank = (
            relevant_ranks[0] if relevant_ranks else None
        )

        for k in K_VALUES:

            top_k_sources = retrieved_sources[:k]

            relevant_count = sum(
                source == expected_source
                for source in top_k_sources
            )

            # Precision@K: relevant retrieved chunks / K
            precision = relevant_count / k

            # Recall@K: whether the expected source was retrieved
            recall = 1.0 if relevant_count > 0 else 0.0

            # Reciprocal rank of the first relevant result
            reciprocal_rank = (
                1 / first_relevant_rank
                if first_relevant_rank is not None
                and first_relevant_rank <= k
                else 0.0
            )

            detailed_results.append({
                "question": question,
                "expected_source": expected_source,
                "k": k,
                "retrieved_sources": " | ".join(top_k_sources),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank
            })

    # Calculate average metrics for each K
    detailed_df = pd.DataFrame(detailed_results)

    for k in K_VALUES:

        subset = detailed_df[detailed_df["k"] == k]

        summary_results.append({
            "k": k,
            "mean_precision_at_k": subset["precision_at_k"].mean(),
            "mean_recall_at_k": subset["recall_at_k"].mean(),
            "mean_reciprocal_rank": subset["reciprocal_rank"].mean()
        })

    summary_df = pd.DataFrame(summary_results)

    # Save results
    detailed_path = RESULTS_DIR / "retrieval_detailed_results.csv"
    summary_path = RESULTS_DIR / "retrieval_metrics_summary.csv"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")
    print(summary_df.round(4).to_string(index=False))

    print("\nDetailed results saved to:", detailed_path)
    print("Metric summary saved to:", summary_path)


if __name__ == "__main__":
    evaluate_retrieval()
