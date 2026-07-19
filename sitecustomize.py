from __future__ import annotations

import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
_cache_root = _repo_root / ".cache" / "huggingface"

for _name, _path in {
    "XDG_CACHE_HOME": _repo_root / ".cache",
    "HF_HOME": _cache_root,
    "HF_DATASETS_CACHE": _cache_root / "datasets",
    "HF_HUB_CACHE": _cache_root / "hub",
    "HF_EVALUATE_CACHE": _cache_root / "evaluate",
    "HF_METRICS_CACHE": _cache_root / "metrics",
    "HF_MODULES_CACHE": _cache_root / "modules",
}.items():
    os.environ.setdefault(_name, str(_path))
