from pathlib import Path

import chromadb

from src.ingestion import extract_pdf
from src.chunking import create_chunks
from src.embeddings import generate_embeddings
from src.vector_store import store_embeddings, DB_PATH, COLLECTION_NAME


def process_document(pdf_path):
    """
    Extract, chunk, embed and store an uploaded PDF.

    Existing documents with the same filename are replaced.
    """

    pdf_path = Path(pdf_path)

    # Step 1: Extract PDF pages
    pages = extract_pdf(pdf_path)

    if not any(page["text"] for page in pages):
        raise ValueError("The PDF contains no extractable text.")

    # Step 2: Create text chunks
    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    # Step 3: Generate embeddings
    embedded_chunks = generate_embeddings(chunks)

    # Step 4: Find existing chunks for this document
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing = collection.get(
        where={"document_name": pdf_path.name},
        include=["metadatas"]
    )

    # Remove previous version of this document
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Step 5: Store the new embeddings
    collection = store_embeddings(embedded_chunks)

    return {
        "document_name": pdf_path.name,
        "pages_processed": len(pages),
        "chunks_created": len(embedded_chunks),
        "total_chunks_stored": collection.count()
    }