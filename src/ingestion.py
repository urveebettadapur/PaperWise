import pymupdf
from pathlib import Path


def extract_pdf(pdf_path):
    """Extract text and metadata from each page of a PDF."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file is not a PDF.")

    extracted_pages = []

    try:
        with pymupdf.open(pdf_path) as document:

            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            total_pages = document.page_count

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if not text:
                    print(
                        f"Warning: No extractable text on "
                        f"page {page_number} of {pdf_path.name}"
                    )

                page_data = {
                    "document_name": pdf_path.name,
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "text": text
                }

                extracted_pages.append(page_data)

    except pymupdf.FileDataError as error:
        raise ValueError(f"Unable to read PDF: {error}") from error

    return extracted_pages


def extract_pdfs(folder_path):
    """Extract text from every PDF in a folder."""

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    pdf_files = sorted(
        file for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the folder.")

    all_pages = []

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = extract_pdf(pdf_file)
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    pdf_folder = Path("data/sample_papers")

    all_pages = extract_pdfs(pdf_folder)

    print("\nPAPERWISE - PDF INGESTION")
    print("-" * 40)

    document_names = sorted(set(
        page["document_name"] for page in all_pages
    ))

    print(f"Documents processed: {len(document_names)}")
    print(f"Total pages extracted: {len(all_pages)}")

    print("\nDOCUMENT SUMMARY")

    for document_name in document_names:
        document_pages = [
            page for page in all_pages
            if page["document_name"] == document_name
        ]

        print(f"{document_name}: {len(document_pages)} pages")

    print("\nSAMPLE EXTRACTED TEXT")
    print("-" * 40)

    first_page = all_pages[0]

    print(f"Document: {first_page['document_name']}")
    print(f"Page: {first_page['page_number']}")
    print(first_page["text"][:500])