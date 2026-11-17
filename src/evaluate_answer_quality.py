
from pathlib import Path
import json
import pandas as pd

from langchain_ollama import ChatOllama


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "results" / "answer_evaluation_results.csv"
OUTPUT_DIR = PROJECT_DIR / "results"


def evaluate_answer_quality():

    data = pd.read_csv(INPUT_PATH)

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        format="json"
    )

    results = []

    for index, row in data.iterrows():

        print(f"\nEvaluating answer {index + 1}/{len(data)}")
        print(row["question"])

        evaluation_prompt = f"""
You are a fair and objective evaluator of a RAG chatbot.

Evaluate the answer using the evidence and scoring rules below.

QUESTION:
{row['question']}

REFERENCE ANSWER:
{row['reference_answer']}

RETRIEVED CONTEXT:
{row['retrieved_context']}

GENERATED ANSWER:
{row['generated_answer']}

SCORING RULES:

1. FAITHFULNESS (0 to 1):
Measure whether the factual claims in the generated answer
are supported by the retrieved context.

1.0 = All factual claims are supported.
0.7 = Main claims are supported, with minor unsupported detail.
0.4 = Some claims are supported, but significant claims are unsupported.
0.0 = The answer is entirely unsupported or contradicts the context.

2. ANSWER CORRECTNESS (0 to 1):
Measure whether the generated answer correctly answers the question
and agrees with the reference answer.

1.0 = Correct answer, even if phrased differently or more concisely.
0.7 = Main answer is correct but misses an important detail.
0.4 = Partially correct, with significant missing or incorrect information.
0.0 = Does not answer the question or gives an incorrect answer.

IMPORTANT:
- Do not require the generated answer to copy the reference answer word for word.
- Do not give zero merely because the answer is concise.
- Do not penalize omission of information that is not necessary to answer the question.
- Do not penalize a correct answer for lacking unrelated policy details.
- Check the retrieved context before judging faithfulness.
- If the generated answer says the same fact as the reference answer,
  award correctness credit.
- Never invent a reason for reducing a score.

Return only valid JSON:
{{
  "faithfulness": 0.0,
  "answer_correctness": 0.0,
  "reason": "Brief explanation of both scores"
}}
"""

        try:
            response = llm.invoke(evaluation_prompt)
            scores = json.loads(response.content)

            faithfulness = float(scores["faithfulness"])
            correctness = float(scores["answer_correctness"])

            if not (0 <= faithfulness <= 1):
                raise ValueError("Faithfulness score outside 0-1.")

            if not (0 <= correctness <= 1):
                raise ValueError("Correctness score outside 0-1.")

            reason = str(scores.get("reason", ""))

        except Exception as error:

            print("Evaluation error:", error)

            faithfulness = None
            correctness = None
            reason = "Evaluation failed; manual review required."

        results.append({
            "question": row["question"],
            "generated_answer": row["generated_answer"],
            "faithfulness": faithfulness,
            "answer_correctness": correctness,
            "evaluation_reason": reason
        })

        print("Faithfulness:", faithfulness)
        print("Answer correctness:", correctness)
        print("Reason:", reason)

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "answer_quality_results.csv"
    results_df.to_csv(output_path, index=False)

    valid_results = results_df.dropna(
        subset=["faithfulness", "answer_correctness"]
    )

    print("\n========== ANSWER QUALITY SUMMARY ==========")

    if not valid_results.empty:

        print(
            "Average faithfulness:",
            round(valid_results["faithfulness"].mean(), 3)
        )

        print(
            "Average answer correctness:",
            round(valid_results["answer_correctness"].mean(), 3)
        )

        print(
            "Successfully evaluated:",
            len(valid_results),
            "of",
            len(results_df)
        )

    else:
        print("No valid scores. Manual review is required.")

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    evaluate_answer_quality()
