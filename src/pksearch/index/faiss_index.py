from __future__ import annotations

import json
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class DenseIndex:
    index: hnswlib.Index
    chunk_ids: List[str]
    doc_ids: List[str]
    titles: List[str]
    texts: List[str]
    dim: int


def embed_chunks(
    chunks_path: Path,
    model_name: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[str], float]:
    model = SentenceTransformer(model_name)

    chunk_ids: List[str] = []
    doc_ids: List[str] = []
    titles: List[str] = []
    texts: List[str] = []

    for r in read_jsonl(chunks_path):
        chunk_ids.append(r["chunk_id"])
        doc_ids.append(r["doc_id"])
        titles.append(r.get("metadata", {}).get("title", ""))
        texts.append(r.get("text", "") or "")

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dt = time.time() - t0
    return emb.astype(np.float32), chunk_ids, doc_ids, titles, texts, dt


def build_hnsw_index(embeddings: np.ndarray, ef_construction: int = 200, M: int = 16) -> hnswlib.Index:
    dim = embeddings.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=M)
    idx.add_items(embeddings, np.arange(embeddings.shape[0]))
    idx.set_ef(50)
    return idx


def save_dense(index: DenseIndex, path: Path) -> None:
    """
    Save HNSW separately + metadata as pickle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    hnsw_path = path.with_suffix(".hnsw")
    meta_path = path.with_suffix(".meta.pkl")

    index.index.save_index(str(hnsw_path))
    meta = {
        "chunk_ids": index.chunk_ids,
        "doc_ids": index.doc_ids,
        "titles": index.titles,
        "texts": index.texts,
        "dim": index.dim,
    }
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)


def load_dense(path: Path) -> DenseIndex:
    hnsw_path = path.with_suffix(".hnsw")
    meta_path = path.with_suffix(".meta.pkl")
    if not hnsw_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing dense index files: {hnsw_path} / {meta_path}")

    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    idx = hnswlib.Index(space="cosine", dim=int(meta["dim"]))
    idx.load_index(str(hnsw_path))
    idx.set_ef(50)

    return DenseIndex(
        index=idx,
        chunk_ids=meta["chunk_ids"],
        doc_ids=meta["doc_ids"],
        titles=meta["titles"],
        texts=meta["texts"],
        dim=int(meta["dim"]),
    )


def search_dense(didx: DenseIndex, query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    labels, distances = didx.index.knn_query(query_emb, k=top_k)
    labels = labels[0]
    distances = distances[0]

    out = []
    for lbl, dist in zip(labels, distances):
        i = int(lbl)
        out.append(
            {
                "chunk_id": didx.chunk_ids[i],
                "doc_id": didx.doc_ids[i],
                "score": float(1.0 - dist),
                "title": didx.titles[i],
                "text": didx.texts[i],
            }
        )
    return out
