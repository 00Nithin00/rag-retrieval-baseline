# PKSearch: A Reproducible Retrieval Baseline for RAG Systems

## 1. Motivation

Retrieval-Augmented Generation (RAG) systems rely heavily on retrieval quality. 
However, many implementations lack:

- Reproducible evaluation
- Baseline comparisons (sparse vs dense vs hybrid)
- Deterministic pipelines
- Clear measurement of Recall and MRR

This project establishes a reproducible retrieval baseline with evaluation-driven development.

The primary goal of Week 1 was to:

> Build a measurable retrieval system and evaluation harness before adding generation.

---

## 2. Dataset

I use the **SciFact** dataset from the BEIR benchmark suite.

- Documents: 5,183
- Queries (test split): 300
- Relevance labels: document-level (qrels)

BEIR ensures standardized IR evaluation.

---

## 3. System Architecture

Pipeline:

```

Raw Dataset
↓
Canonical Parsing (docs.jsonl)
↓
Chunking (fixed / paragraph)
↓
Indexing
├── BM25 (rank-bm25)
└── Dense (MiniLM + HNSW)
↓
Hybrid Fusion (RRF)
↓
Evaluation (Recall@10, MRR@10)

````

---

## 4. Chunking Strategy

### Fixed Chunking
- 350 words per chunk
- 60 word overlap
- Stable ID format: `doc_id::0003`

### Paragraph Chunking
- Paragraph-based merging up to 350 words

All chunk outputs are deterministic.

---

## 5. Retrieval Methods

### 5.1 Sparse Retrieval (BM25)

- Library: `rank-bm25`
- Custom deterministic tokenizer
- Indexed chunks
- Persisted index

---

### 5.2 Dense Retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Normalized embeddings (cosine similarity)
- Index: HNSW (`hnswlib`)
- Embeddings cached to disk

---

### 5.3 Hybrid Retrieval

Reciprocal Rank Fusion (RRF):

score(d) = Σ_i 1 / (k + rank_i(d))

- Retrieve top 50 from BM25
- Retrieve top 50 from Dense
- Fuse into final top 10

---

## 6. Evaluation Methodology

Evaluation is performed using BEIR qrels (document-level).

Procedure:

1. Retrieve top-k chunks
2. Map chunk → document ID
3. Deduplicate document IDs while preserving rank
4. Compute:

- Recall@10
- MRR@10

---

## 7. Results (SciFact Test Split)

| System | Recall@10 | MRR@10 | Avg Latency (ms) |
|--------|----------:|-------:|-----------------:|
| BM25   | 0.7557    | 0.5985 | 16.20 |
| Dense  | 0.7789    | 0.5997 | 31.17 |
| Hybrid (RRF) | **0.7964** | **0.6336** | 33.39 |

### Observations

- Dense retrieval improves Recall over BM25.
- Hybrid retrieval improves both Recall and MRR.
- Sparse retrieval remains competitive at lower latency.
- Hybrid provides best overall retrieval quality.

---

## 8. Reproducibility

### 8.1 Environment

Python 3.11+

Install dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
````

---

### 8.2 Download Dataset

```bash
python run.py -m pksearch.ingest.download --dataset scifact
```

---

### 8.3 Parse and Chunk

```bash
python run.py -m pksearch.ingest.parse --dataset scifact
python run.py -m pksearch.ingest.chunk --dataset scifact --chunker fixed
```

---

### 8.4 Build Indexes

```bash
python run.py -m pksearch.index.build_bm25 --dataset scifact --chunker fixed
python run.py -m pksearch.index.build_dense --dataset scifact --chunker fixed
```

---

### 8.5 Run Evaluation

```bash
python run.py -m pksearch.eval.run_eval \
  --dataset scifact \
  --split test \
  --chunker fixed \
  --top-k 10

Note: The project currently uses a lightweight launcher (`run.py`)
to ensure the src/ layout resolves correctly across environments.
Future revisions will expose proper console entry points.

```

---

## 9. Limitations (Week 1 Scope)

* No reranking
* No metadata weighting
* No query routing
* No faithfulness evaluation
* No generation module

Week 1 establishes the baseline retrieval layer only.

---

## 10. Future Work

* Structure-aware chunking
* Metadata indexing
* Query classification and routing
* Cross-encoder reranking
* Citation-first generation
* Faithfulness evaluation

---

## 11. Key Principles

* Evaluation before optimization
* Deterministic pipelines
* Baselines before complexity
* Measure improvements
