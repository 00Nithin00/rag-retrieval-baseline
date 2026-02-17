from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    indexes_dir: Path = data_dir / "indexes"
    eval_dir: Path = data_dir / "eval"


settings = Settings()
