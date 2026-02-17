from __future__ import annotations

import json
from pathlib import Path
import typer
from tqdm import tqdm

from pksearch.config import settings

app = typer.Typer(add_completion=False)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@app.command()
def main(
    dataset: str = typer.Option("scifact", help="BEIR dataset name"),
):
    raw_dir = settings.raw_dir / dataset
    corpus_path = raw_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    out_dir = settings.processed_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docs.jsonl"

    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in tqdm(read_jsonl(corpus_path), desc=f"Parsing {dataset} corpus"):
            # BEIR corpus schema (typical): {"_id": "...", "title": "...", "text": "..."}
            doc_id = rec.get("_id")
            if doc_id is None:
                raise ValueError("Corpus record missing _id")

            title = rec.get("title") or ""
            text = rec.get("text") or ""

            doc = {
                "doc_id": str(doc_id),
                "title": title,
                "text": text,
                "metadata": {
                    "source": "beir",
                    "dataset": dataset,
                    "created_at": None,
                    "url": None,
                },
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1

    typer.echo(f"[OK] Wrote {n} docs to {out_path}")


if __name__ == "__main__":
    app()
