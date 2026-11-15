"""Training from a YAML recipe (configs/train/*.yaml) with Ultralytics; the run is tracked in runs/<name>/run.json."""

import json
from pathlib import Path

from .data import manifest
from .tracking import record


def load_config(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def train(cfg_path, out="runs", device=None, **override):
    """Trains, evaluates the best weights on val and records the run; returns run.json as a dict."""
    from ultralytics import YOLO

    from . import evaluate

    cfg = load_config(cfg_path)
    args = {**cfg["train"], **override}
    data = Path(cfg["data"])
    man = data.parent / "manifest.json"
    data_info = json.loads(man.read_text()) if man.exists() else manifest.build(data.parent)
    res = YOLO(cfg["model"]).train(
        data=str(data),
        project=str(Path(out).resolve()),
        name=cfg["name"],
        exist_ok=True,
        seed=cfg.get("seed", 0),
        device=device,
        **args,
    )
    run_dir = Path(res.save_dir)
    best = run_dir / "weights" / "best.pt"
    metrics = evaluate.dataset(best, data.parent, **cfg.get("eval", {}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return record(
        run_dir,
        dict(cfg, train=args),
        metrics=metrics,
        data={k: data_info.get(k) for k in ("name", "version", "sha256", "splits")},
        weights=[best],
    )
