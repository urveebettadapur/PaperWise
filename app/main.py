import streamlit as st
import tempfile
from pathlib import Path
import chromadb

from app.document_processor import process_document
from src.generation import generate_answer
from src.vector_store import DB_PATH, COLLECTION_NAME


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PaperWise | Research Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = set()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📚 PaperWise")

    st.caption("Your AI-powered research assistant")

    st.divider()

    st.subheader("Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:

            for uploaded_file in uploaded_files:

                st.write(f"Processing: {uploaded_file.name}")

                try:

                    with tempfile.TemporaryDirectory() as temp_dir:

                        pdf_path = Path(temp_dir) / uploaded_file.name
                        pdf_path.write_bytes(uploaded_file.getvalue())

                        with st.spinner(
                            f"Processing {uploaded_file.name}..."
                        ):
                            result = process_document(pdf_path)

                    st.success(
                        f"Processed {result['document_name']} "
                        f"({result['chunks_created']} chunks)"
                    )

                except Exception as error:

                    st.error(
                        f"Failed to process {uploaded_file.name}: {error}"
                    )

    st.divider()

    st.subheader("Indexed Documents")

    collection = get_collection()

    stored_data = collection.get(
        include=["metadatas"]
    )

    document_names = sorted({
        metadata["document_name"]
        for metadata in stored_data["metadatas"]
    })

    if document_names:

        for document in document_names:
            st.write(f"📄 {document}")

    else:
        st.info("No documents indexed yet.")

    st.caption(f"Total indexed chunks: {collection.count()}")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.title("PaperWise Research Assistant")

st.write(
    "Ask questions about your research papers and "
    "get answers grounded in their content."
)

st.divider()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"- **{source['document']}** "
                        f"— Page {source['page']}"
                    )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your research papers..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching papers and generating an answer..."):

            try:

                result = generate_answer(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("View Sources", expanded=True):

                        for source in sources:

                            st.markdown(
                                f"- **{source['document']}** "
                                f"— Page {source['page']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                error_message = (
                    f"Something went wrong while generating the answer: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })