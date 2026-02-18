import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import runpy

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        mod = sys.argv[2]
        sys.argv = sys.argv[2:] 
        runpy.run_module(mod, run_name="__main__")
    else:
        raise SystemExit("Usage: python run.py -m <module> [args...]")
