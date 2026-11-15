"""Run tracking: runs/<name>/run.json (code commit, package versions, config, dataset version, metrics, weight hashes);
also logged to MLflow when MLFLOW_TRACKING_URI is set and mlflow is installed."""
import json
import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

from . import __version__
from .registry import sha256

PACKAGES = ("hwdet", "ultralytics", "torch", "numpy", "opencv-python-headless", "hwsynth", "ds-page-generation")


def git_state(cwd=None):
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def record(run_dir, config, metrics=None, data=None, weights=(), **extra):
    """Writes run_dir/run.json and returns it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(hwdet=__version__, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git=git_state(), python=platform.python_version(),
               packages=versions(), config=config, data=data, metrics=metrics or {},
               weights={Path(w).name: sha256(w) for w in weights if Path(w).exists()}, **extra)
    (run_dir / "run.json").write_text(json.dumps(rec, indent=1, default=str))
    _mlflow(rec, run_dir)
    return rec


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def _mlflow(rec, run_dir):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=Path(run_dir).name):
        mlflow.log_params({k: str(v)[:250] for k, v in rec["config"].items()})
        mlflow.set_tags({"git_commit": rec["git"]["commit"], "data_sha": (rec["data"] or {}).get("sha256", "")})
        mlflow.log_metrics(dict(_flat(rec["metrics"])))
        mlflow.log_artifact(str(Path(run_dir) / "run.json"))
