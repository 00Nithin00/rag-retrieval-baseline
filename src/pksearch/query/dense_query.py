from __future__ import annotations

import time
import numpy as np
import typer
from sentence_transformers import SentenceTransformer

from pksearch.config import settings
from pksearch.index.faiss_index import load_dense, search_dense

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
    query: str = typer.Option(...),
    top_k: int = typer.Option(10),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
):
    idx_path = settings.indexes_dir / dataset / f"dense_{chunker}"
    didx = load_dense(idx_path)

    model = SentenceTransformer(model_name)

    t0 = time.time()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    emb_ms = (time.time() - t0) * 1000

    t1 = time.time()
    results = search_dense(didx, q, top_k=top_k)
    search_ms = (time.time() - t1) * 1000

    print(f"\nQuery: {query}")
    print(f"Embed latency: {emb_ms:.2f} ms | Search latency: {search_ms:.2f} ms")
    print("-" * 90)

    for r in results:
        snippet = (r["text"][:220] + "…") if len(r["text"]) > 220 else r["text"]
        print(f'{r["chunk_id"]}  sim={r["score"]:.3f}  doc_id={r["doc_id"]}')
        if r.get("title"):
            print(f'  title: {r["title"]}')
        print(f"  snippet: {snippet}\n")


if __name__ == "__main__":
    app()
