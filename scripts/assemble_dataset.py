"""Wrapper so dataset assembly runs without installing the package.

    python scripts/assemble_dataset.py --audio-dir examples/audio --out data/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
