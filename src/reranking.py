from sentence_transformers import CrossEncoder
from src.retrieval import search_papers_hybrid


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load the model only once
cross_encoder = None


def get_reranking_model():
    global cross_encoder

    if cross_encoder is None:
        print("Loading cross-encoder model...")
        cross_encoder = CrossEncoder(MODEL_NAME)

    return cross_encoder


def rerank_results(query, results, top_k=5):
    """
    Rerank retrieved passages using a cross-encoder.
    """

    if not results:
        return []

    model = get_reranking_model()

    # Create query-passage pairs
    pairs = [
        [query, result["document"]]
        for result in results
    ]

    # Calculate relevance scores
    scores = model.predict(
        pairs,
        show_progress_bar=False
    )

    # Attach reranking scores
    reranked_results = []

    for result, score in zip(results, scores):

        reranked_result = result.copy()
        reranked_result["rerank_score"] = float(score)

        reranked_results.append(reranked_result)

    # Sort by cross-encoder relevance
    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


if __name__ == "__main__":

    question = "What is the attention mechanism used in Transformer models?"

    print("\nPAPERWISE - CROSS-ENCODER RERANKING")
    print("-" * 45)

    # Retrieve a larger candidate set using hybrid search
    candidates = search_papers_hybrid(
        question,
        top_k=20,
        candidate_k=20
    )

    print(f"\nCandidates retrieved: {len(candidates)}")

    # Rerank the candidates
    results = rerank_results(
        question,
        candidates,
        top_k=5
    )

    print("\nTOP RERANKED RESULTS")
    print("-" * 45)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 30)

        print(f"Document: {result['metadata']['document_name']}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Reranking Score: {result['rerank_score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")

        print("\nRetrieved Text:")
        print(result["document"][:500])