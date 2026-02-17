from __future__ import annotations

import time
import typer

from pksearch.config import settings
from pksearch.index.faiss_index import embed_chunks, build_hnsw_index, DenseIndex, save_dense

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    batch_size: int = typer.Option(64),
):
    chunks_path = settings.processed_dir / dataset / f"chunks_{chunker}.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    out_path = settings.indexes_dir / dataset / f"dense_{chunker}"

    emb, chunk_ids, doc_ids, titles, texts, emb_time = embed_chunks(
        chunks_path=chunks_path,
        model_name=model_name,
        batch_size=batch_size,
    )

    t1 = time.time()
    hnsw = build_hnsw_index(embeddings=emb)
    build_time = time.time() - t1

    didx = DenseIndex(index=hnsw, chunk_ids=chunk_ids, doc_ids=doc_ids, titles=titles, texts=texts, dim=emb.shape[1])
    save_dense(didx, out_path)

    typer.echo(f"[OK] Dense index built: {out_path}.hnsw + {out_path}.meta.pkl")
    typer.echo(f"[STATS] chunks={len(chunk_ids)} dim={emb.shape[1]} emb_time_sec={emb_time:.2f} build_time_sec={build_time:.2f}")
    typer.echo(f"[INFO] model={model_name}")


if __name__ == "__main__":
    app()
