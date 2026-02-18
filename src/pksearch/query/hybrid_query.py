from __future__ import annotations

import time
import numpy as np
import typer
from sentence_transformers import SentenceTransformer

from pksearch.config import settings
from pksearch.index.bm25 import load_index as load_bm25, search as bm25_search
from pksearch.index.faiss_index import load_dense, search_dense
from pksearch.index.hybrid import rrf_fuse

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed"),
    query: str = typer.Option(...),
    top_k: int = typer.Option(10),
    bm25_k: int = typer.Option(50),
    dense_k: int = typer.Option(50),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
):
    bm25_path = settings.indexes_dir / dataset / f"bm25_{chunker}.pkl"
    dense_path = settings.indexes_dir / dataset / f"dense_{chunker}"

    bm25 = load_bm25(bm25_path)
    didx = load_dense(dense_path)

    t0 = time.time()
    bm25_res = bm25_search(bm25, query=query, top_k=bm25_k)
    bm25_ms = (time.time() - t0) * 1000

    model = SentenceTransformer(model_name)
    t1 = time.time()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    emb_ms = (time.time() - t1) * 1000

    t2 = time.time()
    dense_res = search_dense(didx, q, top_k=dense_k)
    dense_ms = (time.time() - t2) * 1000

    t3 = time.time()
    fused = rrf_fuse(bm25_res, dense_res, k=60, top_k=top_k)
    fuse_ms = (time.time() - t3) * 1000

    print(f"\nQuery: {query}")
    print(f"BM25: {bm25_ms:.2f} ms | Embed: {emb_ms:.2f} ms | Dense: {dense_ms:.2f} ms | Fuse: {fuse_ms:.2f} ms")
    print("-" * 90)

    for r in fused:
        snippet = (r["text"][:220] + "…") if len(r["text"]) > 220 else r["text"]
        print(f'{r["chunk_id"]}  rrf={r["rrf_score"]:.6f}  doc_id={r["doc_id"]}')
        if r.get("title"):
            print(f'  title: {r["title"]}')
        print(f"  snippet: {snippet}\n")


if __name__ == "__main__":
    app()
