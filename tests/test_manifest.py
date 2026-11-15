"""Dataset manifest: counts and a content hash that changes with any label."""
import json

import cv2
import numpy as np

from hwdet.data import manifest


def make(ds, labels):
    for split, rows in labels.items():
        for sub in ("images", "labels", "meta"):
            (ds / split / sub).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows):
            cv2.imwrite(str(ds / split / "images" / f"{i}.jpg"), np.full((32, 24), 255, np.uint8))
            (ds / split / "labels" / f"{i}.txt").write_text(r)
            (ds / split / "meta" / f"{i}.json").write_text(json.dumps(dict(kind="form", lang="en")))


def test_counts_hash_and_yaml(tmp_path):
    make(tmp_path, {"train": ["0 0.5 0.5 0.2 0.1\n1 0.5 0.2 0.3 0.1\n", "1 0.5 0.5 0.2 0.1\n"], "val": [""]})
    m = manifest.build(tmp_path, "toy")
    tr = m["splits"]["train"]
    assert tr["pages"] == 2 and tr["boxes"] == {"handwriting": 1, "print": 2} and tr["pages_without_handwriting"] == 1
    assert tr["kinds"] == {"form": 2} and m["splits"]["val"]["boxes"] == {}
    assert "train: train/images" in (tmp_path / "data.yaml").read_text()
    (tmp_path / "train/labels/1.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    assert manifest.build(tmp_path)["sha256"] != m["sha256"]
