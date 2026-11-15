"""Detection-set validation and packing."""

import json
import zipfile

import cv2
import numpy as np

from hwdet.data import pack as dspack


def make(tmp, pages):
    for split, name, kind, doc, rows in pages:
        for sub in ("images", "labels", "meta"):
            (tmp / split / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp / split / "images" / f"{name}.jpg"), np.full((1024, 768), 200, np.uint8))
        (tmp / split / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n")
        (tmp / split / "meta" / f"{name}.json").write_text(json.dumps(dict(lang="en", kind=kind, doc=doc)))
    return tmp


PAGES = [
    ("train", "a", "notes", "d1", ["0 0.5 0.5 0.2 0.1", "1 0.3 0.3 0.1 0.05"]),
    ("train", "b", "print", "d2", ["1 0.5 0.5 0.2 0.1"]),
    ("val", "c", "hand", "d3", ["0 0.5 0.5 0.4 0.4", "0 0.2 0.2 0.1 0.1"]),
]


def test_a_good_set_validates_packs_loads_and_is_not_compressed(tmp_path):
    ds = make(tmp_path / "ds", PAGES)
    assert dspack.validate(ds) == {}
    index = dspack.pack(ds, tmp_path / "out", shard=1)
    assert [s["file"] for s in index["shards"]] == ["train_000.npz", "train_001.npz", "val_000.npz"]
    for s in index["shards"]:
        with zipfile.ZipFile(tmp_path / "out" / s["file"]) as z:
            assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    z = dspack.load(tmp_path / "out" / "val_000.npz")
    assert z["images"].shape == (1, 1024, 768) and list(z["names"]) == ["c"] and len(z["labels"][0]) == 2
    assert np.allclose(z["labels"][0][1], [0, 0.2, 0.2, 0.1, 0.1]) and list(z["kind"]) == ["hand"]


def test_the_check_finds_what_is_wrong(tmp_path):
    bad = PAGES + [
        ("train", "e", "notes", "d1", ["0 1.2 0.5 0.1 0.1"]),
        ("val", "f", "form", "d1", ["1 0.5 0.5 0.1 0.1"]),
        ("train", "g", "hand", "d9", ["0 0.95 0.5 0.2 0.1"]),
    ]
    found = dspack.validate(make(tmp_path / "ds", bad))
    assert set(found) == {"outside 0..1", "box leaves the page", "no handwriting box", "document in both splits"}
    assert found["document in both splits"] == ["d1"]
