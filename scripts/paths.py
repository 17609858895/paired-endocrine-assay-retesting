"""Shared locations for original inputs and generated tables."""
import os
from pathlib import Path

WORK = Path(os.environ.get("ENDOCRINE_WORKDIR", "work")).resolve()
DATA = WORK / "data"
RESULTS = WORK / "results"
EVIDENCE = Path(os.environ.get("CALUX_INPUTS", WORK / "sources")).resolve()
for directory in [DATA, RESULTS]:
    directory.mkdir(parents=True, exist_ok=True)
