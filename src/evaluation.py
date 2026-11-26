from src.retrieval import (
    search_papers,
    search_papers_bm25,
    search_papers_hybrid
)
from src.reranking import rerank_results


# Evaluation metrics

def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids, k=5):
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


# Retrieval methods

def retrieve_chunk_ids(method, query, k=5, candidate_k=20):

    if method == "Semantic":
        results = search_papers(query, top_k=k)
        return results["ids"][0]

    elif method == "BM25":
        results = search_papers_bm25(query, top_k=k)
        return [result["chunk_id"] for result in results]

    elif method == "Hybrid":
        results = search_papers_hybrid(
            query,
            top_k=k,
            candidate_k=candidate_k
        )
        return [result["chunk_id"] for result in results]

    elif method == "Hybrid + Reranking":
        candidates = search_papers_hybrid(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k
        )

        results = rerank_results(
            query,
            candidates,
            top_k=k
        )

        return [result["chunk_id"] for result in results]

    else:
        raise ValueError(f"Unknown retrieval method: {method}")


# Evaluate one question using one method

def evaluate_method(method, query, relevant_ids, k=5, candidate_k=20):

    retrieved_ids = retrieve_chunk_ids(
        method,
        query,
        k,
        candidate_k
    )

    return {
        "precision_at_k": precision_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "recall_at_k": recall_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "mrr": reciprocal_rank(
            retrieved_ids, relevant_ids, k
        ),
        "retrieved_ids": retrieved_ids
    }


# Evaluation dataset


EVALUATION_DATASET = [
    {
        "question": "What is the attention mechanism used in Transformer models?",
        "relevant_chunks": {
            "attention_is_all_you_need.pdf_page_5_chunk_2",
            "attention_is_all_you_need.pdf_page_5_chunk_3",
            "attention_is_all_you_need.pdf_page_2_chunk_8",
            "attention_is_all_you_need.pdf_page_3_chunk_1",
            "attention_is_all_you_need.pdf_page_10_chunk_4"
        }
    },
    {
        "question": "How does Dense Passage Retrieval retrieve relevant passages?",
        "relevant_chunks": {
            "dense_passage_retrieval.pdf_page_2_chunk_10",
            "dense_passage_retrieval.pdf_page_1_chunk_2",
            "dense_passage_retrieval.pdf_page_2_chunk_11",
            "dense_passage_retrieval.pdf_page_2_chunk_3"
        }
    },
    {
        "question": "What is the lost-in-the-middle problem in language models?",
        "relevant_chunks": {
            "lost_in_the_middle.pdf_page_7_chunk_3",
            "lost_in_the_middle.pdf_page_5_chunk_5",
            "lost_in_the_middle.pdf_page_2_chunk_5",
            "lost_in_the_middle.pdf_page_1_chunk_7",
            "lost_in_the_middle.pdf_page_2_chunk_6"
        }
    },
    {
        "question": "What is Retrieval-Augmented Generation and why is it useful?",
        "relevant_chunks": {
            "rag_knowledge_intensive.pdf_page_1_chunk_3",
            "rag_knowledge_intensive.pdf_page_2_chunk_4",
            "rag_knowledge_intensive.pdf_page_5_chunk_10"
        }
    }
]


# Evaluate all questions and compare methods

def evaluate_dataset(dataset, k=5, candidate_k=20):

    methods = [
        "Semantic",
        "BM25",
        "Hybrid",
        "Hybrid + Reranking"
    ]

    results = {}

    for method in methods:

        print(f"\nEvaluating: {method}")

        question_results = []

        for item in dataset:

            result = evaluate_method(
                method,
                item["question"],
                item["relevant_chunks"],
                k,
                candidate_k
            )

            question_results.append(result)

        # Calculate macro averages across questions
        results[method] = {
            "precision_at_k": sum(
                r["precision_at_k"] for r in question_results
            ) / len(question_results),

            "recall_at_k": sum(
                r["recall_at_k"] for r in question_results
            ) / len(question_results),

            "mrr": sum(
                r["mrr"] for r in question_results
            ) / len(question_results),

            "question_results": question_results
        }

    return results


# Main execution

if __name__ == "__main__":

    results = evaluate_dataset(
        EVALUATION_DATASET,
        k=5,
        candidate_k=20
    )

    print("\n\nPAPERWISE - MULTI-QUESTION RETRIEVAL EVALUATION")
    print("=" * 75)

    print(f"\nNumber of evaluation questions: {len(EVALUATION_DATASET)}")

    print("\nAVERAGE EVALUATION RESULTS")
    print("-" * 75)

    print(
        f"{'Method':<25}"
        f"{'Precision@5':<17}"
        f"{'Recall@5':<15}"
        f"{'MRR@5':<12}"
    )

    print("-" * 75)

    for method, metrics in results.items():

        print(
            f"{method:<25}"
            f"{metrics['precision_at_k']:<17.4f}"
            f"{metrics['recall_at_k']:<15.4f}"
            f"{metrics['mrr']:<12.4f}"
        )

    print("\n\nQUESTION-WISE RESULTS")
    print("=" * 75)

    for index, item in enumerate(EVALUATION_DATASET):

        print(f"\nQuestion {index + 1}: {item['question']}")
        print("-" * 75)

        for method, metrics in results.items():

            question_result = metrics["question_results"][index]

            print(
                f"{method:<25}"
                f"P@5: {question_result['precision_at_k']:.2f}  "
                f"R@5: {question_result['recall_at_k']:.2f}  "
                f"MRR@5: {question_result['mrr']:.2f}"
            )