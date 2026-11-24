import ollama

from src.ingestion import extract_pdfs
from src.chunking import create_chunks


EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 32


def generate_embeddings(chunks, batch_size=BATCH_SIZE):
    """
    Generate embeddings for text chunks using Ollama.

    Returns chunks with their corresponding embedding vectors.
    """

    embedded_chunks = []

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise ValueError("Embedding count does not match chunk count.")

        for chunk, embedding in zip(batch, embeddings):

            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding

            embedded_chunks.append(embedded_chunk)

        print(
            f"Embedded {len(embedded_chunks)}/{len(chunks)} chunks"
        )

    return embedded_chunks


if __name__ == "__main__":

    print("\nPAPERWISE - EMBEDDING GENERATION")
    print("-" * 40)

    pages = extract_pdfs("data/sample_papers")
    chunks = create_chunks(pages)

    embedded_chunks = generate_embeddings(chunks)

    print("\nEMBEDDING SUMMARY")
    print("-" * 40)

    print(f"Chunks processed: {len(embedded_chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    if embedded_chunks:
        sample = embedded_chunks[0]

        print(f"Vector dimensions: {len(sample['embedding'])}")
        print(f"Sample chunk ID: {sample['chunk_id']}")
        print(f"Sample vector: {sample['embedding'][:10]}")