import chromadb

from src.ingestion import extract_pdfs
from src.chunking import create_chunks
from src.embeddings import generate_embeddings


DB_PATH = "chroma_db"
COLLECTION_NAME = "paperwise_research"


def store_embeddings(embedded_chunks):
    """
    Store text chunks, embeddings and metadata in ChromaDB.
    """

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [
        chunk["chunk_id"]
        for chunk in embedded_chunks
    ]

    documents = [
        chunk["text"]
        for chunk in embedded_chunks
    ]

    embeddings = [
        chunk["embedding"]
        for chunk in embedded_chunks
    ]

    metadatas = [
        {
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in embedded_chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return collection


if __name__ == "__main__":

    print("\nPAPERWISE - CHROMADB STORAGE")
    print("-" * 40)

    # Step 1: Extract PDF pages
    pages = extract_pdfs("data/sample_papers")

    # Step 2: Create text chunks
    chunks = create_chunks(pages)

    # Step 3: Generate embeddings
    embedded_chunks = generate_embeddings(chunks)

    # Step 4: Store everything in ChromaDB
    collection = store_embeddings(embedded_chunks)

    print("\nSTORAGE SUMMARY")
    print("-" * 40)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database location: {DB_PATH}")
    print(f"Chunks submitted: {len(embedded_chunks)}")
    print(f"Chunks stored: {collection.count()}")

    # Verify a stored record
    sample = collection.get(
        limit=1,
        include=["documents", "metadatas"]
    )

    if sample["ids"]:
        print("\nSAMPLE STORED RECORD")
        print("-" * 40)

        print(f"ID: {sample['ids'][0]}")
        print(f"Metadata: {sample['metadatas'][0]}")
        print(f"Text: {sample['documents'][0][:300]}")