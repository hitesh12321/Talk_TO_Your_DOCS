# TtoYD — Talk to Your Docs

A self-correcting, document-grounded Q&A chatbot built around **Corrective RAG (CRAG)** principles. Instead of blindly trusting vector similarity search, TtoYD validates retrieved context with an LLM grader and automatically retries with a rewritten query when retrieval fails — before ever generating an answer.

---

## Why this exists

Naive RAG pipelines treat retrieval as a solved step: embed the query, pull the top-k chunks, generate an answer. The problem is that *embedding-similar* is not the same as *actually relevant* — and a system that can't tell the difference will confidently answer from the wrong context.

TtoYD adds a validation layer on top of retrieval: every retrieved chunk is graded for relevance before it's allowed to reach the generation step, and if nothing relevant is found, the query is rewritten and retrieval is retried — up to a capped number of attempts — before falling back to generation regardless.

---

## Architecture

The query pipeline is built with **LangGraph** as a cyclic state machine rather than a linear chain, since the core requirement — validate, retry, then generate — needs conditional branching and loops that a simple chain can't express.

```
START
  │
  ▼
retrieve ──────────────────────────────┐
  │                                    │
  ▼                                    │
grade_documents                        │
  │                                    │
  ├── ≥1 relevant chunk ──────► generate ──► END
  │
  └── 0 relevant chunks
      AND retry_count < 2
          │
          ▼
      transform_query
      (rewrite query,
       retry_count += 1)
          │
          └──────► loops back to retrieve
```

| Node | Responsibility |
|---|---|
| `retrieve` | Embeds the user's question and pulls the top-k chunks from the FAISS index |
| `grade_documents` | An LLM evaluates each retrieved chunk individually for true relevance to the question — not just embedding proximity |
| `transform_query` | Rewrites the question for better retrieval when no relevant chunks are found; increments `retry_count` |
| `generate` | Produces the final answer, grounded only in the chunks that passed grading |

A shared state object flows through every node:

```python
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int
```

Retries are capped at **2** to guarantee the cycle terminates and to avoid diminishing-returns latency — if the document genuinely doesn't contain the answer, no amount of query rewriting will manufacture it.

---

## Indexing pipeline

Before any querying happens, documents are processed once into a searchable index:

```
PDF
 │
 ▼
PdfReader          — raw text extraction (pypdf) with custom regex cleanup
                      for paragraph breaks vs. line-wrap artifacts
 │
 ▼
Chunker             — RecursiveCharacterTextSplitter
                      (500 char chunks, 100 char overlap)
 │
 ▼
LocalEmbedding      — HuggingFace bge-small-en-v1.5, run locally
 │
 ▼
VectorStore (FAISS) — index built and persisted to ./vector_db
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** | Enables conditional branching and cycles — needed for the retry/self-correction loop |
| Vector search | **FAISS** | Local, dependency-light similarity search; no external DB needed for single-corpus retrieval |
| Embeddings | **HuggingFace (`bge-small-en-v1.5`)** | Runs locally, no external API cost or rate limits for embedding |
| LLM | **Groq (Llama 3.3 70B)**, fallback **Gemini 2.0 Flash** | Fast inference via Groq; automatic fallback keeps the app usable without a Groq key |
| PDF parsing | **pypdf** (custom wrapper) | Lightweight extraction with custom whitespace/paragraph normalization |
| UI | **Streamlit** | Fast interactive interface for upload, query, and inspecting retrieved chunks |

---

## Project structure

```
TtoYD/
├── src/
│   ├── pdf_reader.py       # PDF text extraction + cleaning
│   ├── textchunker.py      # Recursive chunking into Document objects
│   ├── local_embedding.py  # HuggingFace embedding wrapper
│   ├── vector_store.py     # FAISS index build / save / load / search
│   ├── indexing.py         # End-to-end indexing pipeline
│   └── rag_graph.py        # LangGraph query pipeline (retrieve → grade → generate)
├── streamlitUI.py          # Streamlit front-end
├── vector_db/               # Persisted FAISS index (generated, not checked in)
└── README.md
```

---

## Setup

```bash
# clone
git clone https://github.com/<your-username>/TtoYD.git
cd TtoYD

# install dependencies
pip install -r requirements.txt

# set environment variables
# create a .env file with:
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here   # optional fallback
```

## Usage

**Run the Streamlit app:**
```bash
streamlit run streamlitUI.py
```

1. Upload a PDF and click **Create Index** — this builds and saves a local FAISS index
2. Ask a question in the query box
3. View the generated answer along with the exact chunks used to ground it

**Or run indexing / retrieval directly from the command line:**
```bash
python -m src.indexing      # build the index from a PDF
python -m src.retrival      # interactive CLI query loop
```

---

## Design notes & known limitations

- **Distance metric**: FAISS is currently used with its default Euclidean (L2) distance. Cosine similarity — generally better suited for sentence embeddings — can be enabled by normalizing embeddings and setting `distance_strategy=DistanceStrategy.COSINE`, but requires rebuilding the index from scratch.
- **Grading threshold**: generation proceeds as soon as even one retrieved chunk passes relevance grading, rather than requiring a majority. This is a deliberately lenient threshold worth revisiting for stricter grounding.
- **Per-chunk grading**: each retrieved chunk is graded with its own LLM call rather than batched, which is simple but adds latency — a production version would likely batch grading into a single structured-output call.
- **Storage**: FAISS is a local, file-based similarity search library, not a hosted vector database — it has no native multi-user concurrency or metadata filtering. A planned improvement is migrating to **Qdrant** for native metadata filtering and concurrent access at scale.

---

## Roadmap

- [ ] Migrate vector storage from FAISS to Qdrant for metadata filtering and multi-user support
- [ ] Switch to cosine similarity with normalized embeddings
- [ ] Batch document grading into a single structured LLM call
- [ ] Support multi-document indexing with source-based filtering

---

## Background

This project grew out of trying to understand how retrieval-augmented systems like ChatGPT's context grounding actually work under the hood — specifically, how they avoid confidently answering from irrelevant retrieved context. Rather than reading about the theory alone, this is a from-scratch implementation of the **Corrective RAG (CRAG)** pattern using LangGraph, built to explore where naive RAG breaks down and how a validation-and-retry layer helps.

---

## Author

**Hitesh Saini**
[GitHub](https://github.com/hitesh12321) · [LinkedIn](https://linkedin.com/in/hitesh-saini-b515542a8)
