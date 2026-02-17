from __future__ import annotations

from pathlib import Path
import time
import typer

from pksearch.config import settings
from pksearch.index.bm25 import build_bm25_index, save_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
):
    chunks_path = settings.processed_dir / dataset / (f"chunks_{chunker}.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    out_path = settings.indexes_dir / dataset / f"bm25_{chunker}.pkl"

    t0 = time.time()
    idx = build_bm25_index(chunks_path)
    save_index(idx, out_path)
    dt = time.time() - t0

    typer.echo(f"[OK] BM25 index built: {out_path}")
    typer.echo(f"[STATS] chunks={len(idx.chunk_ids)} build_time_sec={dt:.3f}")


if __name__ == "__main__":
    app()
