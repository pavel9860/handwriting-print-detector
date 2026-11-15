"""Run records and the evaluation helpers that need no model."""

import json

import numpy as np

from hwdet import evaluate, tracking


def test_record_writes_versions_hashes_and_metrics(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    rec = tracking.record(
        tmp_path / "run", dict(lr0=0.001), metrics=dict(map50=0.9), data=dict(version="abc"), weights=[w]
    )
    on_disk = json.loads((tmp_path / "run" / "run.json").read_text())
    assert on_disk["weights"]["best.pt"] == rec["weights"]["best.pt"] and len(rec["weights"]["best.pt"]) == 64
    assert (
        on_disk["packages"]["hwdet"]
        and on_disk["config"] == {"lr0": 0.001}
        and set(on_disk["git"]) == {"commit", "dirty"}
    )


def test_flat_metrics_for_mlflow():
    assert dict(tracking._flat({"a": {"b": 1, "c": "x"}, "d": 2.5, "e": True})) == {"a.b": 1, "d": 2.5}


def test_read_labels_to_pixels(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("0 0.5 0.5 0.5 0.1\n1 0.25 0.25 0.1 0.1\n")
    lab = evaluate.read_labels(f, 200, 100)
    assert np.allclose(lab["handwriting"], [[50, 45, 150, 55]]) and len(lab["print"]) == 1
    assert evaluate.read_labels(tmp_path / "missing.txt", 10, 10)["handwriting"].shape == (0, 4)


def test_report_table():
    s = dict(precision=1.0, recall=0.5, f1=0.667, fp_per_page=0.0, fn_per_page=1.0, pages_ok=0.5)
    out = evaluate.report({"pages": 2, "conf_0.4": {"iou_0.5": {"all": {"handwriting": s}}}})
    assert "| 0.4 | 0.5 | handwriting | 1.000 | 0.500 |" in out
