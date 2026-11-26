# PaperWise

**A smarter way to understand and interact with academic literature.**

PaperWise is an AI-powered research paper assistant that enables users to query academic PDFs and receive context-aware, source-grounded answers. It combines semantic search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking to retrieve relevant research passages before generating answers using a local language model.

## Overview

Reading and navigating multiple research papers can be time-consuming, especially when searching for specific concepts, methodologies, or findings.

PaperWise addresses this by allowing users to upload research papers, ask natural-language questions, and receive answers grounded in the contents of their documents, with references to the original paper and page number.

## Key Features

- **PDF Ingestion:** Extracts text and page-level metadata from academic PDFs.
- **Document Chunking:** Splits extracted text into overlapping chunks for retrieval.
- **Semantic Search:** Retrieves passages based on contextual similarity using vector embeddings.
- **BM25 Keyword Search:** Retrieves passages using keyword relevance.
- **Hybrid Retrieval:** Combines semantic and keyword rankings using Reciprocal Rank Fusion.
- **Cross-Encoder Reranking:** Reorders retrieved passages based on query-document relevance.
- **Context-Grounded Generation:** Generates answers using retrieved document context.
- **Source References:** Provides paper names and page numbers for retrieved information.
- **Interactive Interface:** Supports PDF uploads and conversational querying through Streamlit.
- **Retrieval Evaluation:** Compares four retrieval approaches using Precision@5, Recall@5, and Mean Reciprocal Rank (MRR@5).

## Architecture

```text
              Research Paper PDFs
                       |
                       v
                PDF Ingestion
                       |
                       v
                 Text Chunking
                       |
                       v
               Embedding Generation
                       |
                       v
                 ChromaDB Store
                       |
                       v
                 User Question
                       |
             +---------+---------+
             |                   |
             v                   v
       Semantic Search       BM25 Search
             |                   |
             +---------+---------+
                       |
                       v
              Hybrid Retrieval
                (RRF Fusion)
                       |
                       v
              Cross-Encoder
                 Reranking
                       |
                       v
               Context Assembly
                       |
                       v
             LLM Answer Generation
                       |
                       v
             Answer with Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Application Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Ollama – nomic-embed-text |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 – rank-bm25 |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) |
| Reranking | Sentence Transformers |
| Reranking Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Language Model | Ollama – Llama 3.2 3B |

## Retrieval Evaluation

PaperWise evaluates four retrieval strategies across four research questions using a manually judged relevance dataset.

### Evaluation Results

| Retrieval Method | Precision@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Semantic Search | 0.4500 | 0.5333 | 0.5833 |
| BM25 | 0.3500 | 0.4417 | 0.5833 |
| Hybrid Retrieval | **0.5500** | **0.6667** | 0.5833 |
| Hybrid + Reranking | 0.4500 | 0.5542 | **0.6458** |

### Observations

- Hybrid retrieval achieved the highest Precision@5 and Recall@5 in the evaluation.
- Hybrid + Reranking achieved the highest MRR@5, indicating better positioning of the first relevant result on average.
- Semantic retrieval performed particularly well on the Dense Passage Retrieval question.
- Reranking did not consistently improve the complete top-five result set, demonstrating that reranking quality depends on the initial candidate pool and query.

**Evaluation note:** These results come from a small benchmark of four questions with manually assigned relevance judgments. The candidate pool was formed from the top-five results returned by each retrieval method. Recall therefore reflects the judged candidate pool rather than exhaustive ground truth across every document chunk. Results should be interpreted as preliminary comparative findings.

## Project Structure

```text
PaperWise/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── document_processor.py
│
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── collect_evaluation_candidates.py
│   └── review_evaluation_candidates.py
│
├── data/
│   └── sample_papers/
│
├── docs/
│   └── architecture.md
|
├── inspect_queries.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Install the following before running PaperWise:

- Python 3.12
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/urveebettadapur/PaperWise.git
cd PaperWise
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Ensure Ollama is running before launching the application.

### 5. Run the application

```bash
python -m streamlit run app/main.py
```

Open the local URL displayed in the terminal to access PaperWise.

### 6. Upload research papers

Use the sidebar to upload PDF research papers and process them. Once indexed, enter questions in the chat interface to retrieve relevant information and generate source-grounded answers.

## Evaluation

Run the retrieval evaluation module:

```bash
python -m src.evaluation
```

To regenerate the pooled evaluation candidates:

```bash
python -m src.collect_evaluation_candidates
```

To inspect candidates for a particular evaluation question:

```bash
python -m src.review_evaluation_candidates 1
```

Replace `1` with a question number from 1 to 4.

## Development Utilities

The repository also includes `inspect_queries.py`, a utility script for inspecting the top hybrid retrieval results for selected research questions.

Run it using:

```bash
python inspect_queries.py
```

## Current Scope and Limitations

- The evaluation benchmark currently contains four manually judged questions.
- Retrieval results depend on the quality and coverage of the indexed documents.
- Cross-encoder reranking can only reorder retrieved candidates; it cannot recover passages missing from the initial retrieval pool.
- The application is designed around locally running Ollama models.
- The current implementation focuses on text extraction from PDFs; scanned documents may require OCR support.

## Future Improvements

- Expand the evaluation dataset with more questions and relevance judgments.
- Improve retrieval through query expansion and parameter tuning.
- Add OCR support for scanned research papers.
- Introduce document-level filtering and richer metadata-based search.
- Improve document indexing efficiency and duplicate handling.
- Add automated evaluation and answer-quality metrics.

## Author

**Urvee S Bettadapur**

[GitHub](https://github.com/urveebettadapur)

---

*PaperWise — Making academic literature easier to explore, understand, and navigate.*