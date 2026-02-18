from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import typer
from tqdm import tqdm

from pksearch.config import settings

try:
    import regex as re
except Exception:  
    import re  

app = typer.Typer(add_completion=False)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_word_chunks(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """
    Deterministic word-based chunker.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def paragraph_chunks(text: str, max_words: int) -> List[str]:
    """
    Simple structure-ish chunker:
    - split by blank lines into paragraphs
    - then merge paragraphs until max_words reached
    - if a paragraph is huge, fall back to fixed_word_chunks on it
    """
    text = normalize_ws(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_words = 0

    for p in paras:
        p_words = len(p.split())
        if p_words > max_words * 1.2:
            flush()
            chunks.extend(fixed_word_chunks(p, chunk_words=max_words, overlap_words=int(max_words * 0.15)))
            continue

        if buf_words + p_words <= max_words:
            buf.append(p)
            buf_words += p_words
        else:
            flush()
            buf.append(p)
            buf_words = p_words

    flush()
    return chunks


def stats_from_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def pct(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return float(lengths_sorted[idx])

    return {
        "avg": float(sum(lengths_sorted) / n),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": float(lengths_sorted[-1]),
    }


def build_chunks(
    docs_path: Path,
    chunker: str,
    fixed_chunk_words: int,
    fixed_overlap_words: int,
    para_max_words: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    chunks_out: List[Dict[str, Any]] = []
    chunk_lens_chars: List[int] = []
    n_docs = 0

    for doc in tqdm(read_jsonl(docs_path), desc=f"Chunking ({chunker})"):
        n_docs += 1
        doc_id = doc["doc_id"]
        title = doc.get("title", "")
        text = doc.get("text", "") or ""
        text = normalize_ws(text)

        if chunker == "fixed":
            pieces = fixed_word_chunks(text, chunk_words=fixed_chunk_words, overlap_words=fixed_overlap_words)
        elif chunker == "para":
            pieces = paragraph_chunks(text, max_words=para_max_words)
        else:
            raise ValueError(f"Unknown chunker: {chunker}")

        for i, piece in enumerate(pieces):
            chunk_id = f"{doc_id}::{i:04d}"
            row = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": piece,
                "metadata": {
                    "title": title,
                    "chunk_index": i,
                    "chunker": chunker,
                },
            }
            chunks_out.append(row)
            chunk_lens_chars.append(len(piece))

    stats = {
        "docs": n_docs,
        "chunks": len(chunks_out),
        "chunk_chars": stats_from_lengths(chunk_lens_chars),
    }
    return chunks_out, stats


@app.command()
def main(
    dataset: str = typer.Option("scifact"),
    chunker: str = typer.Option("fixed", help="fixed|para"),
    fixed_chunk_words: int = typer.Option(350, help="fixed chunk size in words"),
    fixed_overlap_words: int = typer.Option(60, help="fixed overlap in words"),
    para_max_words: int = typer.Option(350, help="paragraph chunk max size in words"),
):
    docs_path = settings.processed_dir / dataset / "docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs.jsonl: {docs_path}")

    out_dir = settings.processed_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / ("chunks_fixed.jsonl" if chunker == "fixed" else "chunks_para.jsonl")

    chunks, st = build_chunks(
        docs_path=docs_path,
        chunker=chunker,
        fixed_chunk_words=fixed_chunk_words,
        fixed_overlap_words=fixed_overlap_words,
        para_max_words=para_max_words,
    )

    n = write_jsonl(out_path, chunks)

    typer.echo(f"[OK] Wrote {n} chunks to {out_path}")
    typer.echo(
        f"[STATS] docs={st['docs']} chunks={st['chunks']} "
        f"chars(avg={st['chunk_chars']['avg']:.1f}, p50={st['chunk_chars']['p50']:.0f}, "
        f"p95={st['chunk_chars']['p95']:.0f}, max={st['chunk_chars']['max']:.0f})"
    )


if __name__ == "__main__":
    app()
