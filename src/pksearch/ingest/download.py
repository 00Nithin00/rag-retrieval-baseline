from __future__ import annotations

from pathlib import Path
import typer

from beir.util import download_and_unzip
from pksearch.config import settings

app = typer.Typer(add_completion=False)

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@app.command()
def main(
    dataset: str = typer.Option("scifact", help="BEIR dataset name, e.g., scifact, fiqa, trec-covid"),
    force: bool = typer.Option(False, help="Re-download even if dataset folder exists"),
):
    out_dir: Path = settings.raw_dir / dataset
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        typer.echo(f"[OK] Dataset already present at: {out_dir}")
        raise typer.Exit(code=0)

    url = f"{BEIR_BASE_URL}/{dataset}.zip"
    typer.echo(f"[INFO] Downloading: {url}")
    typer.echo(f"[INFO] To: {settings.raw_dir}")

    download_and_unzip(url, str(settings.raw_dir))

    if not out_dir.exists():
        raise RuntimeError(f"Expected dataset folder not found after download: {out_dir}")

    typer.echo(f"[OK] Download complete: {out_dir}")


if __name__ == "__main__":
    app()
