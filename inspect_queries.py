from src.retrieval import search_papers_hybrid

questions = [
    "How does Dense Passage Retrieval retrieve relevant passages?",
    "What is the lost-in-the-middle problem in language models?",
    "What is Retrieval-Augmented Generation and why is it useful?"
]

for question in questions:
    print("\n\nQUESTION:", question)
    print("-" * 80)

    results = search_papers_hybrid(
        question,
        top_k=8,
        candidate_k=20
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}: {result['chunk_id']}")
        print(result["document"])