import streamlit as st
import faiss
from qa_pipeline import answer_query
from main import load_pickle
import time
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="FiQA - Financial Q&A", page_icon="💰", layout="centered")

# Load resources (cached)
@st.cache_resource
@st.cache_resource
def load_resources():
    if not os.path.exists("faiss_index.index"):
        st.error("Missing file: faiss_index.index")
        st.stop()
    if not os.path.exists("chunks.pkl"):
        st.error("Missing file: chunks.pkl")
        st.stop()
    if not os.path.exists("sources.pkl"):
        st.error("Missing file: sources.pkl")
        st.stop()

    index = faiss.read_index("faiss_index.index")
    chunks = load_pickle("chunks.pkl")
    sources = load_pickle("sources.pkl")
    return index, chunks, sources

index, chunks, sources = load_resources()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom styling
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle {
            font-size: 18px;
            color: #7f8c8d;
        }
        .answer-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            font-size: 18px;
            line-height: 1.6;
        }
        .question {
            font-weight: bold;
            margin-top: 1rem;
            color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💬 FiQA - Ask Financial Questions</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by custom Retrieval-Augmented Generation (RAG)</div>", unsafe_allow_html=True)
st.markdown("---")

# Text input
question = st.text_area("💡 Enter your question below", height=100, placeholder="e.g., What are the current trends in the Indian stock market?")

# Button
if st.button("🔍 Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            answer = answer_query(question, index, chunks, sources)

        # Save to chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})
        st.success("✅ Answer Generated!")

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 🧾 Chat History")
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='question'>❓ {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-box'>{entry['answer']}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")