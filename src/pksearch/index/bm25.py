from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def simple_tokenize(text: str) -> List[str]:
    out = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: List[str]
    doc_ids: List[str]
    titles: List[str]
    texts: List[str]


def build_bm25_index(chunks_path: Path) -> BM25Index:
    chunk_ids: List[str] = []
    doc_ids: List[str] = []
    titles: List[str] = []
    texts: List[str] = []
    tokenized_corpus: List[List[str]] = []

    for r in read_jsonl(chunks_path):
        chunk_ids.append(r["chunk_id"])
        doc_ids.append(r["doc_id"])
        titles.append(r.get("metadata", {}).get("title", ""))
        txt = r.get("text", "") or ""
        texts.append(txt)
        tokenized_corpus.append(simple_tokenize(txt))

    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(
        bm25=bm25,
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        titles=titles,
        texts=texts,
    )


def save_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)


def search(index: BM25Index, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    q_tokens = simple_tokenize(query)
    scores = np.asarray(index.bm25.get_scores(q_tokens), dtype=np.float32)
    if scores.size == 0:
        return []

    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for i in top_idx:
        results.append(
            {
                "chunk_id": index.chunk_ids[int(i)],
                "doc_id": index.doc_ids[int(i)],
                "score": float(scores[int(i)]),
                "title": index.titles[int(i)],
                "text": index.texts[int(i)],
            }
        )
    return results
