"""Run provenance: manifests that make a run reproducible and auditable.

Every stage writes a JSON manifest containing the configuration used, the
package versions, the git revision and the timestamp. Two runs of the same stage
should differ only in the timestamp, which is exactly the property the README's
"making analysis steps reproducible" item asks for.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn

from .config import PROJECT_ROOT


def git_revision(short=True):
    """Current git commit hash, or ``None`` outside a git checkout."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else [
            "git", "rev-parse", "HEAD"]
        out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_dirty():
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return None


def environment_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": _version("matplotlib"),
        "seaborn": _version("seaborn"),
        "statsmodels": _version("statsmodels"),
    }


def _version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def write_manifest(path, stage, config=None, extra=None):
    """Write a JSON manifest describing how an output directory was produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "environment": environment_info(),
    }
    if config is not None:
        payload["config"] = asdict(config) if is_dataclass(config) else dict(config)
    if extra:
        payload["extra"] = _jsonable(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (numpy.integer, numpy.floating)):
        return obj.item()
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    return obj