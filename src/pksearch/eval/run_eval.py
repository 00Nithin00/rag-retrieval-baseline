from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import typer
from sentence_transformers import SentenceTransformer

from pksearch.config import settings
from pksearch.eval.metrics import recall_at_k, mrr_at_k
from pksearch.index.bm25 import load_index as load_bm25, search as bm25_search
from pksearch.index.faiss_index import load_dense, search_dense
from pksearch.index.hybrid import rrf_fuse

app = typer.Typer(add_completion=False)


def read_queries(path: Path) -> Dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[str(r["_id"])] = r["text"]
    return out


def read_qrels_tsv(path: Path, min_rel: int = 1) -> Dict[str, Set[str]]:
    """
    BEIR qrels TSV schema usually: query-id, corpus-id, score
    Sometimes 4 columns with an unused "0": query-id, 0, corpus-id, score
    We handle both.
    """
    qrels: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        # If header is present, it might be: query-id corpus-id score
        # We'll detect non-numeric score and treat as header.
        if header:
            def is_int(x: str) -> bool:
                try:
                    int(x)
                    return True
                except Exception:
                    return False
            # If last column isn't int, it's header; else it's a valid row, process it
            if not is_int(header[-1]):
                pass
            else:
                row = header
                rows = [row] + list(reader)
                reader = rows  # type: ignore

        if isinstance(reader, list):
            rows_iter = reader
        else:
            rows_iter = reader

        for row in rows_iter:
            if not row:
                continue
            if len(row) == 3:
                qid, docid, score = row
            elif len(row) >= 4:
                qid, _, docid, score = row[0], row[1], row[2], row[3]
            else:
                continue

            try:
                s = int(score)
            except Exception:
                continue
            if s < min_rel:
                continue
            qrels.setdefault(str(qid), set()).add(str(docid))
    return qrels


def chunks_to_ranked_docs(results: List[Dict[str, Any]], k_docs: int = 10) -> List[str]:
    """
    Convert ranked chunks to ranked doc_ids, deduping doc_ids while preserving order.
    """
    seen = set()
    ranked = []
    for r in results:
        did = str(r["doc_id"])
        if did in seen:
            continue
        seen.add(did)
        ranked.append(did)
        if len(ranked) >= k_docs:
            break
    return ranked


def eval_system(
    system: str,
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    top_k: int,
    bm25=None,
    didx=None,
    model=None,
) -> Tuple[float, float, float, int]:
    recall_sum = 0.0
    mrr_sum = 0.0
    latency_sum_ms = 0.0
    n = 0

    for qid, qtext in queries.items():
        rel = qrels.get(qid)
        if not rel:
            continue

        t0 = time.time()

        if system == "bm25":
            chunk_res = bm25_search(bm25, qtext, top_k=top_k)
        elif system == "dense":
            qemb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            chunk_res = search_dense(didx, qemb, top_k=top_k)
        elif system == "hybrid":
            bm = bm25_search(bm25, qtext, top_k=50)
            qemb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            de = search_dense(didx, qemb, top_k=50)
            chunk_res = rrf_fuse(bm, de, k=60, top_k=top_k)
        else:
            raise ValueError(f"Unknown system: {system}")

        latency_ms = (time.time() - t0) * 1000.0
        ranked_docs = chunks_to_ranked_docs(chunk_res, k_docs=top_k)

        recall_sum += recall_at_k(ranked_docs, rel, k=top_k)
        mrr_sum += mrr_at_k(ranked_docs, rel, k=top_k)
        latency_sum_ms += latency_ms
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0

    return recall_sum / n, mrr_sum / n, latency_sum_ms / n, n


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    split: str = typer.Option("test", help="train|dev|test"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
    top_k: int = typer.Option(10),
    max_queries: int = typer.Option(0, help="0 = all queries; else evaluate first N (for quick debug)"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
):
    raw = settings.raw_dir / dataset
    queries_path = raw / "queries.jsonl"
    qrels_path = raw / "qrels" / f"{split}.tsv"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels: {qrels_path}")

    queries = read_queries(queries_path)
    qrels = read_qrels_tsv(qrels_path, min_rel=1)

    # keep only queries that appear in qrels
    filtered = [(qid, q) for (qid, q) in queries.items() if qid in qrels]
    if max_queries and max_queries > 0:
        filtered = filtered[:max_queries]
    queries = dict(filtered)

    bm25_path = settings.indexes_dir / dataset / f"bm25_{chunker}.pkl"
    dense_path = settings.indexes_dir / dataset / f"dense_{chunker}"

    bm25 = load_bm25(bm25_path)
    didx = load_dense(dense_path)
    model = SentenceTransformer(model_name)

    systems = ["bm25", "dense", "hybrid"]
    rows = []
    for sysname in systems:
        r, m, lat, n = eval_system(
            sysname, queries, qrels, top_k=top_k, bm25=bm25, didx=didx, model=model
        )
        rows.append((sysname, r, m, lat, n))

    print(f"\nDataset: {dataset} | Split: {split} | Chunker: {chunker} | top_k={top_k} | queries={len(queries)}")
    print("-" * 78)
    print(f"{'System':<10} {'Recall@10':>10} {'MRR@10':>10} {'AvgLat(ms)':>12} {'N':>6}")
    print("-" * 78)
    for sysname, r, m, lat, n in rows:
        print(f"{sysname:<10} {r:>10.4f} {m:>10.4f} {lat:>12.2f} {n:>6d}")
    print("-" * 78)


if __name__ == "__main__":
    app()
