import streamlit as st
import tempfile
import os
from src.pdf_reader import PdfReader
from src.textchunker import Chunker
from src.vector_store import VectorStore
from src.indexing import build_index
from src.rag_graph import app

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="TALK TO YOUR DOCS",
    page_icon="📄",
    layout="centered"
)

st.title("TALK TO YOUR DOCS📄")
st.markdown("---")

# -----------------------------
# Session State
# -----------------------------
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "store" not in st.session_state:
    st.session_state.store = None

if "results" not in st.session_state:
    st.session_state.results = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("Upload PDF")

uploaded_file = st.file_uploader(
    "Choose PDF",
    type=["pdf"]
)

# -----------------------------
# Create Index
# -----------------------------
if st.button("Create Index", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Building Vector Index..."):

            # Save uploaded pdf temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            try:
                store = build_index(pdf_path, source=uploaded_file.name)

                st.session_state.store = store
                st.session_state.index_ready = True
                st.session_state.results = []
                st.session_state.answer = ""

                st.success(f"✅ Index created for '{uploaded_file.name}'")

            except Exception as e:
                st.error(f"Failed to build index: {e}")

            finally:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

st.markdown("---")

# -----------------------------
# Ask Question
# -----------------------------
st.subheader("Ask Question")

query = st.text_input(
    "Question",
    placeholder="Ask something about the document...",
    label_visibility="collapsed"
)

if st.button("Search", use_container_width=True):

    if not st.session_state.index_ready:
        st.warning("Please create the index first.")

    elif query.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Running LangGraph Query Pipeline..."):

            try:
                # LangGraph Pipeline execution
                inputs = {"question": query, "retry_count": 0}
                output = app.invoke(inputs)

                st.session_state.answer = output.get("generation", "")
                st.session_state.results = output.get("documents", [])
            except Exception as e:
                st.error(f"Error during query execution: {e}")

st.markdown("---")

# -----------------------------
# Results & Answer Display
# -----------------------------
if st.session_state.answer:

    st.subheader("💡 AI Answer")
    st.success(st.session_state.answer)

if st.session_state.results:

    st.subheader("📚 Validated Context Chunks")

    for i, doc in enumerate(st.session_state.results, start=1):

        with st.expander(f"Chunk {i}"):

            st.write(doc.page_content)

            if "chunk_id" in doc.metadata:
                st.caption(f"Chunk ID : {doc.metadata['chunk_id']}")


            