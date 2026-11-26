# PaperWise — System Architecture

## 1. Architecture Overview

PaperWise follows a Retrieval-Augmented Generation (RAG) architecture. It retrieves relevant passages from indexed research papers and supplies them as context to a language model to generate source-grounded answers.

The architecture consists of five main stages:

1. Document ingestion and preprocessing
2. Embedding generation and vector storage
3. Hybrid information retrieval
4. Cross-encoder reranking and answer generation
5. Retrieval evaluation

## 2. System Architecture Diagram

```mermaid
flowchart TD
    A[Research Paper PDFs] --> B[PDF Ingestion - PyMuPDF]
    B --> C[Page-Level Text Extraction]
    C --> D[Text Chunking - LangChain]
    D --> E[Embedding Generation - Ollama]
    E --> F[(ChromaDB Vector Store)]

    G[User Question] --> H[Streamlit Interface]
    H --> I[Retrieval Pipeline]

    F --> J[Semantic Search]
    I --> J
    I --> K[BM25 Keyword Search]

    J --> L[Reciprocal Rank Fusion]
    K --> L

    L --> M[Cross-Encoder Reranking]
    M --> N[Context Assembly]
    N --> O[LLM Answer Generation - Llama 3.2]
    O --> P[Answer with Source References]
    P --> H

    Q[Evaluation Dataset] --> R[Retrieval Evaluation]
    F --> R
    R --> S[Precision@5, Recall@5, MRR@5]
```

## 3. Document Processing Pipeline

### 3.1 PDF Ingestion

- PyMuPDF extracts text from uploaded research papers.
- Text is extracted page by page.
- Document names and page numbers are retained as metadata.

**Module:** `src/ingestion.py`

### 3.2 Text Chunking

- Extracted page text is divided into smaller, overlapping chunks.
- Chunk size: 500 characters.
- Chunk overlap: 100 characters.
- Each chunk retains its document name, page number and chunk index.

**Module:** `src/chunking.py`

### 3.3 Embedding Generation

- Each chunk is converted into a numerical vector representation.
- Ollama's `nomic-embed-text` model generates 768-dimensional embeddings.
- Embeddings are generated in batches.

**Module:** `src/embeddings.py`

### 3.4 Vector Storage

- ChromaDB stores the document chunks, their embeddings and associated metadata.
- Cosine similarity is used for semantic retrieval.

**Module:** `src/vector_store.py`

## 4. Retrieval Pipeline

PaperWise combines two complementary retrieval approaches.

### 4.1 Semantic Search

- Converts the user query into an embedding.
- Compares the query embedding against stored document embeddings.
- Retrieves passages based on contextual similarity.

### 4.2 BM25 Keyword Search

- Tokenizes the indexed document chunks.
- Uses BM25 scoring to identify passages containing relevant terms.
- Complements semantic retrieval by matching important keywords.

### 4.3 Hybrid Retrieval — Reciprocal Rank Fusion

The rankings from semantic search and BM25 are combined using Reciprocal Rank Fusion (RRF).

The RRF score is calculated as:

\[
RRF(d) = \sum_{r \in R(d)} \frac{1}{k + rank_r(d)}
\]

Where:

- \(d\) represents a retrieved document chunk.
- \(R(d)\) represents the retrieval rankings containing that chunk.
- \(rank_r(d)\) is the chunk's position in a particular ranking.
- \(k\) is a constant set to 60.

The combined ranking helps leverage both contextual similarity and keyword relevance.

**Module:** `src/retrieval.py`

## 5. Reranking and Answer Generation

### 5.1 Cross-Encoder Reranking

- The hybrid retrieval stage supplies candidate passages.
- A cross-encoder evaluates each query-passage pair.
- Candidates are reordered according to their relevance scores.
- The highest-ranked passages are selected for answer generation.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Module:** `src/reranking.py`

### 5.2 Context Assembly

The selected passages are assembled into a context containing:

- Source document name
- Page number
- Retrieved passage text

This allows the language model to generate answers grounded in the indexed research papers.

### 5.3 Answer Generation

- The retrieved context and user question are passed to the language model.
- Ollama runs the `llama3.2:3b` model locally.
- The model generates an answer using the supplied context.
- Source document names and page references are returned alongside the answer.

**Module:** `src/generation.py`

## 6. Application Layer

PaperWise uses Streamlit to provide an interactive interface.

The application supports:

- Uploading research paper PDFs.
- Processing and indexing uploaded documents.
- Viewing indexed documents and chunk counts.
- Asking natural-language questions.
- Displaying generated answers and their source references.

**Modules:**
- `app/main.py`
- `app/document_processor.py`

## 7. Retrieval Evaluation

PaperWise includes an evaluation pipeline to compare four retrieval approaches:

1. Semantic Search
2. BM25
3. Hybrid Retrieval
4. Hybrid Retrieval with Cross-Encoder Reranking

The evaluation uses four research questions and manually assigned relevance judgments.

The following metrics are calculated:

| Metric | Purpose |
|---|---|
| Precision@5 | Measures the proportion of relevant results among the top five retrieved chunks. |
| Recall@5 | Measures the proportion of judged relevant chunks retrieved in the top five. |
| MRR@5 | Measures how highly the first relevant result is ranked. |

**Modules:**
- `src/evaluation.py`
- `src/collect_evaluation_candidates.py`
- `src/review_evaluation_candidates.py`

The benchmark is preliminary and uses a pooled candidate set rather than exhaustive relevance judgments across all indexed chunks.

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain |
| Embedding Model | nomic-embed-text |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Reranking | Sentence Transformers |
| Reranking Model | MiniLM Cross-Encoder |
| Language Model | Llama 3.2 3B |
| Local Model Runtime | Ollama |

---

*PaperWise combines semantic retrieval, keyword matching and reranking to make academic research easier to explore and understand.*