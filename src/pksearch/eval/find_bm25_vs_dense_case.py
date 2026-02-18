from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Set, List, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from pksearch.config import settings
from pksearch.index.bm25 import load_index as load_bm25, search as bm25_search
from pksearch.index.faiss_index import load_dense, search_dense


def read_queries(path: Path) -> Dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[str(r["_id"])] = r["text"]
    return out


def read_qrels_tsv(path: Path, min_rel: int = 1) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    def is_int(x: str) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    start = 0
    if rows and rows[0] and not is_int(rows[0][-1]):
        start = 1

    for row in rows[start:]:
        if not row:
            continue
        if len(row) == 3:
            qid, docid, score = row
        else:
            qid, _, docid, score = row[0], row[1], row[2], row[3]
        try:
            s = int(score)
        except Exception:
            continue
        if s < min_rel:
            continue
        qrels.setdefault(str(qid), set()).add(str(docid))
    return qrels


def chunks_to_ranked_docs(results: List[Dict[str, Any]], k_docs: int = 10) -> List[str]:
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


def pretty_top(results: List[Dict[str, Any]], k: int = 5) -> str:
    lines = []
    for i, r in enumerate(results[:k], start=1):
        snippet = (r["text"][:160] + "…") if len(r["text"]) > 160 else r["text"]
        lines.append(
            f"{i:>2}. doc_id={r['doc_id']} chunk_id={r['chunk_id']} score={r.get('score', 0):.4f}\n"
            f"    title: {r.get('title','')}\n"
            f"    snippet: {snippet}"
        )
    return "\n".join(lines)


def main(dataset: str = "scifact", split: str = "test", chunker: str = "fixed"):
    raw = settings.raw_dir / dataset
    queries_path = raw / "queries.jsonl"
    qrels_path = raw / "qrels" / f"{split}.tsv"

    queries = read_queries(queries_path)
    qrels = read_qrels_tsv(qrels_path, min_rel=1)

    bm25 = load_bm25(settings.indexes_dir / dataset / f"bm25_{chunker}.pkl")
    didx = load_dense(settings.indexes_dir / dataset / f"dense_{chunker}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for qid, qtext in queries.items():
        rel = qrels.get(qid)
        if not rel:
            continue

        bm25_chunks = bm25_search(bm25, qtext, top_k=50)
        bm25_docs = chunks_to_ranked_docs(bm25_chunks, k_docs=10)

        qemb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        dense_chunks = search_dense(didx, qemb, top_k=50)
        dense_docs = chunks_to_ranked_docs(dense_chunks, k_docs=10)

        bm1 = bm25_docs[0] if bm25_docs else None
        de1 = dense_docs[0] if dense_docs else None

        if bm1 and de1 and (bm1 in rel) and (de1 not in rel):
            print("\n=== FOUND CASE ===")
            print(f"qid: {qid}")
            print(f"query: {qtext}")
            print(f"relevant_doc_ids (qrels): {sorted(list(rel))[:20]}{' ...' if len(rel) > 20 else ''}")
            print("\nBM25 top-5 chunks:")
            print(pretty_top(bm25_chunks, k=5))
            print("\nDense top-5 chunks:")
            print(pretty_top(dense_chunks, k=5))
            print("\nBM25 ranked doc_ids:", bm25_docs[:10])
            print("Dense ranked doc_ids:", dense_docs[:10])
            return

    print("No case found under this criterion (try another dataset/split/chunker or relax condition).")


if __name__ == "__main__":
    main()
