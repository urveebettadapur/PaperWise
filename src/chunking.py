from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import extract_pdfs


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(pages):
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its document name and page number.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:

        page_text = page["text"].strip()

        if not page_text:
            continue

        text_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(text_chunks, start=1):

            chunk_data = {
                "chunk_id": (
                    f"{page['document_name']}"
                    f"_page_{page['page_number']}"
                    f"_chunk_{chunk_index}"
                ),
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk_text
            }

            chunks.append(chunk_data)

    return chunks


if __name__ == "__main__":

    print("\nPAPERWISE - TEXT CHUNKING")
    print("-" * 40)

    pages = extract_pdfs("data/sample_papers")

    chunks = create_chunks(pages)

    print(f"Documents processed: {len(set(p['document_name'] for p in pages))}")
    print(f"Pages processed: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    print("\nCHUNKS PER DOCUMENT")
    print("-" * 40)

    chunk_counts = Counter(
        chunk["document_name"] for chunk in chunks
    )

    for document, count in sorted(chunk_counts.items()):
        print(f"{document}: {count} chunks")

    print("\nSAMPLE CHUNK")
    print("-" * 40)

    if chunks:
        sample = chunks[0]

        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Document: {sample['document_name']}")
        print(f"Page: {sample['page_number']}")
        print(f"Chunk Index: {sample['chunk_index']}")
        print(f"Characters: {len(sample['text'])}")
        print("\nText:")
        print(sample["text"][:500])