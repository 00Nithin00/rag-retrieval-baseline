from __future__ import annotations

import time
import typer

from pksearch.config import settings
from pksearch.index.bm25 import load_index, search

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
    query: str = typer.Option(..., help="Query text"),
    top_k: int = typer.Option(10),
):
    idx_path = settings.indexes_dir / dataset / f"bm25_{chunker}.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Missing BM25 index: {idx_path}\n"
            f"Build it first:\n"
            f"PYTHONPATH=src python -m pksearch.index.build_bm25 --dataset {dataset} --chunker {chunker}"
        )

    idx = load_index(idx_path)

    t0 = time.time()
    results = search(idx, query=query, top_k=top_k)
    dt_ms = (time.time() - t0) * 1000

    print(f"\nQuery: {query}")
    print(f"Latency: {dt_ms:.2f} ms")
    print("-" * 90)

    for r in results:
        snippet = (r["text"][:220] + "…") if len(r["text"]) > 220 else r["text"]
        print(f'{r["chunk_id"]}  score={r["score"]:.3f}  doc_id={r["doc_id"]}')
        if r.get("title"):
            print(f'  title: {r["title"]}')
        print(f"  snippet: {snippet}\n")


if __name__ == "__main__":
    app()
