"""Make the repository's training/runtime packages importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DEFAULT = Path(__file__).resolve().parents[2]
VISUALNAV_ROOT = Path(os.environ.get("VISUALNAV_ROOT", str(_DEFAULT))).expanduser().resolve()
for path in (VISUALNAV_ROOT / "train", VISUALNAV_ROOT / "diffusion_policy", VISUALNAV_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
