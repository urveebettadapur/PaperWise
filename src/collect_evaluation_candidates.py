import json
from pathlib import Path

from src.evaluation import EVALUATION_DATASET
from src.retrieval import (
    search_papers,
    search_papers_bm25,
    search_papers_hybrid
)
from src.reranking import rerank_results


TOP_K = 5
CANDIDATE_K = 20

METHODS = [
    "Semantic",
    "BM25",
    "Hybrid",
    "Hybrid + Reranking"
]


def collect_candidates(question):

    candidates = {}

    def add_candidate(chunk_id, text, method):

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "retrieved_by": []
            }

        if method not in candidates[chunk_id]["retrieved_by"]:
            candidates[chunk_id]["retrieved_by"].append(method)

    # 1. Semantic retrieval
    semantic_results = search_papers(
        question,
        top_k=TOP_K
    )

    for chunk_id, text in zip(
        semantic_results["ids"][0],
        semantic_results["documents"][0]
    ):
        add_candidate(chunk_id, text, "Semantic")

    # 2. BM25 retrieval
    bm25_results = search_papers_bm25(
        question,
        top_k=TOP_K
    )

    for result in bm25_results:
        add_candidate(
            result["chunk_id"],
            result["document"],
            "BM25"
        )

    # 3. Hybrid retrieval
    hybrid_results = search_papers_hybrid(
        question,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K
    )

    for result in hybrid_results:
        add_candidate(
            result["chunk_id"],
            result["document"],
            "Hybrid"
        )

    # 4. Hybrid + reranking
    hybrid_candidates = search_papers_hybrid(
        question,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K
    )

    reranked_results = rerank_results(
        question,
        hybrid_candidates,
        top_k=TOP_K
    )

    for result in reranked_results:
        add_candidate(
            result["chunk_id"],
            result["document"],
            "Hybrid + Reranking"
        )

    return list(candidates.values())


def main():

    evaluation_candidates = []

    print("PAPERWISE - EVALUATION CANDIDATE COLLECTION")
    print("=" * 60)

    for index, item in enumerate(EVALUATION_DATASET, start=1):

        question = item["question"]

        print(f"\nCollecting candidates for Question {index}...")
        print(question)

        candidates = collect_candidates(question)

        evaluation_candidates.append({
            "question": question,
            "candidates": candidates
        })

        print(f"Unique candidates collected: {len(candidates)}")

    output_path = Path("evaluation_candidates.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_candidates,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETED")
    print(f"Questions processed: {len(evaluation_candidates)}")
    print(f"Output file: {output_path.resolve()}")

    total_candidates = sum(
        len(item["candidates"])
        for item in evaluation_candidates
    )

    print(f"Total unique question-candidate pairs: {total_candidates}")


if __name__ == "__main__":
    main()