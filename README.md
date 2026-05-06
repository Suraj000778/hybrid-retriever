# Hybrid Retriever — BM25 + Vector Search + RRF Fusion

A generic, vendor-independent hybrid retrieval system combining sparse (BM25) and dense (FAISS) retrieval using **Reciprocal Rank Fusion (RRF)**, with optional cross-encoder reranking.

---

## Why Hybrid Retrieval?

| Method | Strength | Weakness |
|---|---|---|
| BM25 (sparse) | Exact keyword matching | Misses semantic meaning |
| Vector search (dense) | Semantic understanding | Misses exact keywords |
| **Hybrid + RRF** | **Best of both** | — |

---

## How It Works

```
Query
  ├── BM25 Search ──────────┐
  │                         ▼
  └── Vector Search ──→ RRF Fusion ──→ (Optional Reranker) ──→ Final Results
```

**RRF Formula:**
```
score(doc) = Σ 1 / (k + rank_i)
```
Where `k=60` (default) and `rank_i` is the document's rank in each result list.

---

## Installation

```bash
pip install rank-bm25 sentence-transformers faiss-cpu numpy
```

---

## Usage

### Basic

```python
from retriever import HybridRetriever

documents = [
    "BM25 is a ranking function used in search engines",
    "FAISS is used for vector similarity search",
    "RAG systems combine retrieval and generation",
]

retriever = HybridRetriever(documents)
results = retriever.retrieve("vector search ranking", top_k=2)

for doc, score in results:
    print(f"{doc} | score: {score}")
```

### With Reranking

```python
retriever = HybridRetriever(documents, use_reranker=True)
results = retriever.retrieve("vector search ranking", top_k=2)
```

### Custom Models

```python
retriever = HybridRetriever(
    documents,
    embedding_model="all-mpnet-base-v2",
    use_reranker=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rrf_k=60,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `documents` | `List[str]` | required | Documents to index |
| `embedding_model` | `str` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `use_reranker` | `bool` | `False` | Enable cross-encoder reranking |
| `reranker_model` | `str` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `rrf_k` | `int` | `60` | RRF fusion constant |

---

## Methods

- `retrieve(query, top_k)` — Main method, returns `List[(doc, score)]`
- `bm25_search(query, top_k)` — BM25 only
- `vector_search(query, top_k)` — Dense search only
- `rrf_fusion(bm25_results, vector_results)` — RRF combination
- `rerank(query, candidates, top_k)` — Cross-encoder reranking

---

## Key Design Decisions

- **Cosine similarity** via `IndexFlatIP` + L2 normalization (better than L2 distance for text)
- **RRF over score normalization** — more robust when BM25 and vector scores are on different scales
- **top_k * 2 candidates** passed to RRF for better fusion coverage
- **Vendor-independent** — works with any document list, no external DB needed
