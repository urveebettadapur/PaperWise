from src.retrieval import search_papers_hybrid


def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Calculate Precision@K.

    Measures the proportion of retrieved chunks
    that are relevant.
    """

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Calculate Recall@K.

    Measures the proportion of all relevant chunks
    that were successfully retrieved.
    """

    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Calculate Reciprocal Rank.

    Measures the rank of the first relevant result.
    """

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def evaluate_retrieval(query, relevant_ids, k=5):
    """
    Evaluate PaperWise hybrid retrieval.
    """

    results = search_papers_hybrid(
        query,
        top_k=20,
        candidate_k=20
    )

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    mrr = reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    return {
    "precision_at_k": precision,
    "recall_at_k": recall,
    "mrr": mrr,
    "retrieved_count": len(retrieved_ids),
    "retrieved_ids": retrieved_ids
    }


if __name__ == "__main__":

    question = (
        "What is the attention mechanism used "
        "in Transformer models?"
    )

    # Manually identified relevant chunks.
    relevant_chunks = {
        "attention_is_all_you_need.pdf_page_2_chunk_4",
        "attention_is_all_you_need.pdf_page_1_chunk_3",
        "attention_is_all_you_need.pdf_page_3_chunk_1",
        "attention_is_all_you_need.pdf_page_10_chunk_4"
    }

    results = evaluate_retrieval(
        question,
        relevant_chunks,
        k=5
    )

    print("\nPAPERWISE - RETRIEVAL EVALUATION")
    print("-" * 45)

    print(f"\nQuestion: {question}")

    print("\nEVALUATION METRICS")
    print("-" * 45)

    print(
        f"Precision@5: "
        f"{results['precision_at_k']:.4f}"
    )

    print(
        f"Recall@5: "
        f"{results['recall_at_k']:.4f}"
    )

    print(
        f"MRR: "
        f"{results['mrr']:.4f}"
    )

    print(
        f"Total retrieved candidates: "
        f"{results['retrieved_count']}"
    )

    print("\nRETRIEVED PASSAGES")
    print("-" * 45)

    for rank, result in enumerate(
        search_papers_hybrid(
            question,
            top_k=5,
            candidate_k=20
        ),
        start=1
    ):
        chunk_id = result["chunk_id"]

        relevance = (
            "Relevant"
            if chunk_id in relevant_chunks
            else "Not marked relevant"
        )

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Relevance: {relevance}")
        print("\nPassage:")
        print(result["document"][:700])