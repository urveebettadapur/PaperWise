import ollama

from src.reranking import rerank_results
from src.retrieval import search_papers_hybrid


GENERATION_MODEL = "llama3.2:3b"


def generate_answer(query, top_k=5, candidate_k=20):
    """
    Generate an answer using retrieved and reranked
    research-paper passages.
    """

    # Step 1: Retrieve candidate passages
    candidates = search_papers_hybrid(
        query,
        top_k=candidate_k,
        candidate_k=candidate_k
    )

    # Step 2: Rerank the retrieved passages
    results = rerank_results(
        query,
        candidates,
        top_k=top_k
    )

    if not results:
        return {
            "answer": "No relevant research passages were found.",
            "sources": []
        }

    # Step 3: Construct context from the top passages
    context_parts = []
    sources = []

    for index, result in enumerate(results, start=1):

        document_name = result["metadata"]["document_name"]
        page_number = result["metadata"]["page_number"]
        text = result["document"]

        context_parts.append(
            f"[Source {index}]\n"
            f"Paper: {document_name}\n"
            f"Page: {page_number}\n"
            f"Content:\n{text}"
        )

        sources.append({
            "document": document_name,
            "page": page_number
        })

    context = "\n\n".join(context_parts)

    # Step 4: Build the RAG prompt
    prompt = f"""
You are PaperWise, an AI research assistant.

Answer the user's question using ONLY the research
passages provided below.

Instructions:
- Give a clear and accurate explanation.
- Organize the answer into meaningful points.
- Do not introduce information that is not supported
  by the provided passages.
- If the passages do not contain enough information,
  clearly state that.
- Cite the relevant source using its paper name and
  page number.
- Do not invent citations.

Research Context:
{context}

User Question:
{query}

Answer:
"""

    # Step 5: Generate the answer using Ollama
    response = ollama.generate(
        model=GENERATION_MODEL,
        prompt=prompt
    )

    answer = response["response"].strip()

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = (
        "What is the attention mechanism used "
        "in Transformer models?"
    )

    print("\nPAPERWISE - RAG GENERATION")
    print("-" * 45)

    print(f"\nQuestion: {question}")

    result = generate_answer(question)

    print("\nGENERATED ANSWER")
    print("-" * 45)

    print(result["answer"])

    print("\nRETRIEVED SOURCES")
    print("-" * 45)

    for source in result["sources"]:
        print(
            f"Document: {source['document']} | "
            f"Page: {source['page']}"
        )