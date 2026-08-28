# ⚖️ Legal RAG Engine — Contract Analysis & Statutory RAG System

An enterprise-grade, retrieval-augmented generation (RAG) system specialized for **Indian legal statutes, contract compliance auditing, case law, and statutory interpretation**. Built with a hierarchical parent-child chunking architecture, hybrid vector + lexical retrieval, cross-encoder reranking, and NVIDIA LLMs with real-time reasoning streaming.

---

## 🌟 Key Features

- **Hierarchical Parent-Child Chunking**:
  - **Child Chunks (~350 tokens)**: Embedded into vector space (`BAAI/bge-m3`) for high-precision semantic matching without vector bloat.
  - **Parent Chunks (~1800 tokens)**: Preserved to provide full legal context to the LLM during answer generation.
- **Hybrid Dense + Lexical Retrieval**:
  - **Qdrant Vector Database**: Dense vector search across legal embeddings.
  - **BM25 Lexical Search**: Keyword matching for exact statutory terminology and section numbers.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse search results ($k=60$).
- **Legal Metadata-Aware Boosting**: Dynamic relevance multipliers for exact section references (+10.0), statutory act titles (+5.0), domain categories (+3.0), and legal jurisdictions (+2.0).
- **Cross-Encoder Reranking**: Re-scores candidate passages using Cross-Encoder models (`ms-marco-MiniLM-L-6-v2` / `bge-reranker-v2-m3`).
- **Robust PDF Parsing & PyTesseract OCR Fallback**: Native digital PDF extraction via PyMuPDF (`fitz`) with automatic scanned document detection and OCR fallback.
- **NVIDIA LLM Integration & SSE Streaming**: Integrated with `nvidia/nemotron-3-super-120b-a12b`, `openai/gpt-oss-120b`, and `nvidia/nemotron-3-ultra-550b-a55b`. Supports Server-Sent Events (SSE) for streaming reasoning tokens and grounded legal answers.
- **Interactive React + TypeScript Frontend**: Modern glassmorphism UI featuring live legal chat, expandable Evidence Drawers, contract risk analysis, and audit dashboards.
- **100% Test Coverage**: Verified with a comprehensive unit test suite (47/47 passing tests).

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────┐
                               │ Raw Legal Corpus        │
                               │ (PDFs, Markdown Rules)  │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Parser & PyTesseract OCR │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Parent-Child Chunker    │
                               │ (Parent 1800 / Child 350)│
                               └──────┬───────────┬──────┘
                                      │           │
                  Child Chunks Vector │           │ Child Chunks Text
                               ┌──────▼─────┐ ┌───▼────────┐
                               │ Qdrant DB  │ │ BM25 Store │
                               └──────┬─────┘ └───┬────────┘
                                      │           │
                               ┌──────▼───────────▼──────┐
                               │ Hybrid Retriever + RRF  │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Cross-Encoder Reranker  │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Evidence Expansion &    │
                               │ Parent Context Assembly │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ NVIDIA LLM (Nemotron)   │
                               └────────────┬────────────┘
                                            │ SSE Stream
                               ┌────────────▼────────────┐
                               │ React Frontend UI       │
                               └─────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── src/legal_rag/                  # Core Legal RAG Engine Python Package
│   ├── api/                        # FastAPI REST & SSE endpoints
│   ├── chunking/                   # Parent-child clause chunker & token utilities
│   ├── cli/                        # Typer CLI application (`rag`)
│   ├── config.py                   # Centralized Pydantic configuration
│   ├── embedding/                  # BGE-M3 embedding provider
│   ├── engine.py                   # Master LegalRagEngine orchestrator
│   ├── evaluation/                 # Benchmark evaluation runner
│   ├── generation/                 # Grounded generator with exact citation parsing
│   ├── indexing/                   # Qdrant & BM25 index adapters
│   ├── ingestion/                  # Discovery, SHA-256 deduplication & file hashing
│   ├── models/                     # Data models (Document, Chunk, Retrieval, Confidence)
│   ├── parsers/                    # PDF parser, PyTesseract OCR, Markdown parser
│   ├── providers/llm/              # NVIDIA API integration (Nemotron 120B/550B, GPT-OSS 120B)
│   ├── query/                      # Query intent analyzer & cross-reference linker
│   ├── retrieval/                  # Hybrid retriever, RRF blender, confidence scorer
│   └── structure/                  # Statutory section & clause extractor
├── frontend/                       # React + TypeScript + Vite Web Application
│   ├── src/components/             # Chat UI, Evidence Drawer, Risk Analysis components
│   ├── src/pages/                  # Legal Chat, Contracts, Review, Risk, Dashboard
│   └── src/services/ragApi.ts      # API client with SSE streaming support
├── contract_rules/                 # Domain contract rules (employment, lease, nda, sla, vendor)
├── legal_documents/                # PDF/MD corpus across 7 legal domains
├── data/                           # Local storage for Qdrant DB, BM25 indices & artifacts
├── tests/unit/                     # Unit test suite (47 tests)
├── eval_dataset_final.json         # 1,511-line benchmark evaluation dataset
├── docker-compose.yml              # Local Qdrant container manifest
├── pyproject.toml                  # Python package specification
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & **npm**
- **Docker** (for running local Qdrant vector database)
- **Tesseract OCR** (optional, for scanned PDF OCR support)

---

### 1. Backend Setup

1. **Clone the Repository & Create Virtual Environment**:
   ```bash
   git clone https://github.com/Nekilesh001/Legal_rag.git
   cd Legal_rag
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```
   *Set your `NVIDIA_API_KEY` in `.env`.*

4. **Start Qdrant Vector Store**:
   ```bash
   docker-compose up -d
   ```

5. **Run Ingestion (Build Indices)**:
   ```bash
   rag ingest
   ```

6. **Start the FastAPI Backend Server**:
   ```bash
   uvicorn legal_rag.api.main:app --reload --port 8000
   ```
   *The API will be available at `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`).*

---

### 2. Frontend Setup

1. **Navigate to the Frontend Directory**:
   ```bash
   cd frontend
   ```

2. **Install Dependencies & Start Dev Server**:
   ```bash
   npm install
   npm run dev
   ```
   *Open `http://localhost:5173` in your browser to access the Legal Chat UI.*

---

## 💻 CLI Usage (`rag`)

The package includes a rich CLI powered by Typer and Rich:

```bash
# Ingest the corpus
rag ingest

# Query the legal engine from terminal
rag query "What does Section 73 of the Indian Contract Act say regarding damages?"

# Run system evaluation on the benchmark dataset
rag eval --dataset eval_dataset_final.json
```

---

## 🧪 Testing

Run the full unit test suite using `pytest`:

```bash
pytest tests/ -v
```

All 47 unit tests verify parsing, parent-child chunking, vector indexing, RRF blending, confidence scoring, and streaming generation.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
