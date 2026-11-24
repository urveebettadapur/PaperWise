import chromadb
import ollama
import re
from rank_bm25 import BM25Okapi

from src.vector_store import DB_PATH, COLLECTION_NAME
from src.embeddings import EMBEDDING_MODEL


def search_papers(query, top_k=5):
    """
    Search research papers using semantic similarity.
    """

    # Connect to the existing ChromaDB database
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError("The ChromaDB collection is empty.")

    # Generate embedding for the user's question
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    # Retrieve the most relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results



def search_papers_bm25(query, top_k=5):
    """
    Search research papers using BM25 keyword retrieval.
    """

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Retrieve all stored chunks and metadata
    stored_data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]
    ids = stored_data["ids"]

    if not documents:
        raise ValueError("No documents found in ChromaDB.")

    # Tokenize the research-paper chunks
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    tokenized_corpus = [
        tokenize(document)
        for document in documents
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize the user's query
    tokenized_query = tokenize(query)

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank chunks by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "chunk_id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
            "bm25_score": float(scores[index])
        })

    return results


def search_papers_hybrid(query, top_k=5, candidate_k=20):
    """
    Combine semantic and BM25 retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    RRF_K = 60

    # Retrieve candidate chunks from both methods
    semantic_results = search_papers(
        query,
        top_k=candidate_k
    )

    bm25_results = search_papers_bm25(
        query,
        top_k=candidate_k
    )

    # Store combined rankings using chunk IDs
    fused_results = {}

    # Process semantic search ranking
    semantic_ids = semantic_results["ids"][0]
    semantic_documents = semantic_results["documents"][0]
    semantic_metadatas = semantic_results["metadatas"][0]

    for rank, chunk_id in enumerate(semantic_ids, start=1):

        fused_results[chunk_id] = {
            "chunk_id": chunk_id,
            "document": semantic_documents[rank - 1],
            "metadata": semantic_metadatas[rank - 1],
            "rrf_score": 1 / (RRF_K + rank)
        }

    # Process BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk_id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "rrf_score": 0
            }

        fused_results[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    # Sort by combined RRF score
    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]

if __name__ == "__main__":

    print("\nPAPERWISE - SEMANTIC SEARCH")
    print("-" * 40)

    question = (
        "What is the attention mechanism used "
        "in Transformer models?"
    )

    print(f"Question: {question}")

    results = search_papers(question, top_k=5)

    print("\nTOP RETRIEVED RESULTS")
    print("-" * 40)

    for index in range(len(results["ids"][0])):

        print(f"\nResult {index + 1}")
        print("-" * 30)

        print(f"Document: {results['metadatas'][0][index]['document_name']}")
        print(f"Page: {results['metadatas'][0][index]['page_number']}")
        print(f"Chunk ID: {results['ids'][0][index]}")
        print(f"Cosine Distance: {results['distances'][0][index]:.4f}")

        print("\nRetrieved Text:")
        print(results["documents"][0][index][:500])


if __name__ == "__main__":

    question = "What is the attention mechanism used in Transformer models?"

    print("\nPAPERWISE - HYBRID SEARCH (RRF)")
    print("-" * 45)

    results = search_papers_hybrid(
        question,
        top_k=5
    )

    print("\nTOP HYBRID RETRIEVED RESULTS")
    print("-" * 45)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 30)

        print(f"Document: {result['metadata']['document_name']}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Chunk ID: {result['chunk_id']}")

        print("\nRetrieved Text:")
        print(result["document"][:500])