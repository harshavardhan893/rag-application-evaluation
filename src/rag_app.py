
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"


def create_chatbot():

    print("Loading local Qwen model...")

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    print("Loading embedding model...")

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

    policy_prompt = ChatPromptTemplate.from_template("""
You are a company IT support assistant.

Answer the question using ONLY the company policy context below.
Do not use outside knowledge to fill gaps.
If the answer is not available in the context, say:
"I don't have enough information in the company policies."

Context:
{context}

Question:
{question}

Answer:
""")

    general_prompt = ChatPromptTemplate.from_template("""
You are a helpful general-purpose AI assistant.
Answer the user's question clearly and accurately.

Question:
{question}

Answer:
""")

    def ask(question, mode):

        if mode == "Company Policy":

            documents = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in documents
            )

            response = llm.invoke(
                policy_prompt.format(
                    context=context,
                    question=question
                )
            )

            sources = list(dict.fromkeys(
                doc.metadata.get("source", "Unknown")
                for doc in documents
            ))

            return response.content, sources

        elif mode == "General Assistant":

            response = llm.invoke(
                general_prompt.format(question=question)
            )

            return response.content, []

        else:
            raise ValueError("Unknown chatbot mode.")

    return ask


if __name__ == "__main__":

    chatbot = create_chatbot()

    print("\n====================================")
    print("         AI CHATBOT")
    print("====================================")

    while True:

        print("\nChoose a mode:")
        print("1. Company Policy Mode")
        print("2. General Assistant Mode")
        print("Type 'exit' to close.")

        mode_choice = input("\nYour choice: ").strip()

        if mode_choice.lower() in ["exit", "quit"]:
            print("Chatbot closed. Goodbye!")
            break

        if mode_choice == "1":
            mode = "Company Policy"
        elif mode_choice == "2":
            mode = "General Assistant"
        else:
            print("Please enter 1 or 2.")
            continue

        question = input("\nEnter your question: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, sources = chatbot(question, mode)

            print("\nAssistant:", answer)

            if sources:
                print("\nSources:")
                for source in sources:
                    print("-", source)

        except Exception as error:
            print("\nError:", type(error).__name__, "-", error)
