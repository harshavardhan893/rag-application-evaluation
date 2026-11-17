
from pathlib import Path
import sys

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_app import create_chatbot


st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="centered"
)


@st.cache_resource
def load_chatbot():
    return create_chatbot()


st.title("🤖 Company AI Assistant")

st.write(
    "Ask questions about company IT policies "
    "or use the general AI assistant."
)

st.sidebar.title("Chatbot Settings")

mode = st.sidebar.radio(
    "Choose chatbot mode:",
    ["Company Policy", "General Assistant"]
)

if mode == "Company Policy":
    st.sidebar.info(
        "Answers are generated using your company policy documents."
    )
else:
    st.sidebar.info(
        "Ask general questions using the local Qwen model."
    )

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chatbot = load_chatbot()
except Exception as error:
    st.error(f"Could not load chatbot: {error}")
    st.stop()

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.write("📄", source)

# Chat input
question = st.chat_input("Type your question here...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                answer, sources = chatbot(question, mode)

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved sources"):
                        for source in sources:
                            st.write("📄", source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Error: {error}")
