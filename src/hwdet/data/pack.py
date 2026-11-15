"""A detection set (<split>/{images,labels,meta}) checked and packed into uncompressed .npz shards, no pickle"""

import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("handwriting", "print")
SPLITS = ("train", "val")


def items(ds):
    """[(split, name)] of the images of a set, sorted."""
    return sorted((s, p.stem) for s in SPLITS for p in (Path(ds) / s / "images").glob("*.jpg"))


def read(ds, split, name):
    """(grey image, labels (m, 5) float32, meta dict) of one page."""
    d = Path(ds) / split
    img = cv2.imread(str(d / "images" / f"{name}.jpg"), 0)
    rows = [r.split() for r in (d / "labels" / f"{name}.txt").read_text().split("\n") if r.strip()]
    return img, np.array(rows, np.float32).reshape(-1, 5), json.loads((d / "meta" / f"{name}.json").read_text())


def problems(img, lab, meta, shape):
    """What is wrong with one page, as a list of words."""
    out = []
    if img is None or img.shape != shape:
        out.append("size")
    if len(lab) and not np.isin(lab[:, 0], (0, 1)).all():
        out.append("class")
    if len(lab) and not ((lab[:, 1:] >= 0).all() and (lab[:, 1:] <= 1).all()):
        out.append("outside 0..1")
    if len(lab) and not (lab[:, 3:] > 0).all():
        out.append("empty box")
    if len(lab) and not (
        ((lab[:, 1] - lab[:, 3] / 2) >= -1e-3).all()
        and ((lab[:, 1] + lab[:, 3] / 2) <= 1 + 1e-3).all()
        and ((lab[:, 2] - lab[:, 4] / 2) >= -1e-3).all()
        and ((lab[:, 2] + lab[:, 4] / 2) <= 1 + 1e-3).all()
    ):
        out.append("box leaves the page")
    if meta["kind"] != "print" and not (len(lab) and (lab[:, 0] == 0).any()):
        out.append("no handwriting box")
    if meta["kind"] in ("hand", "real_hw") and len(lab) and (lab[:, 0] == 1).any():
        out.append("print box on a handwriting-only page")
    return out


def validate(ds, shape=(1024, 768)):
    """{problem: [names]} over the whole set, plus 'duplicate name' and 'document in both splits'."""
    bad, seen, docs = {}, set(), {s: set() for s in SPLITS}
    for split, name in items(ds):
        img, lab, meta = read(ds, split, name)
        for p in problems(img, lab, meta, shape):
            bad.setdefault(p, []).append(f"{split}/{name}")
        if name in seen:
            bad.setdefault("duplicate name", []).append(name)
        seen.add(name)
        docs[split].add(str(meta["doc"]))
    both = docs["train"] & docs["val"]
    if both:
        bad["document in both splits"] = sorted(both)
    return bad


def pack(ds, out, shard=600, shape=(1024, 768)):
    """Writes the shards and OUT/index.json; returns the index."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"classes": list(CLASSES), "image_shape": list(shape), "shards": []}
    for split in SPLITS:
        names = [n for s, n in items(ds) if s == split]
        for k in range(0, len(names), shard):
            imgs, labs, offs, meta = [], [], [0], []
            for name in names[k : k + shard]:
                img, lab, m = read(ds, split, name)
                imgs.append(img)
                labs.append(lab)
                offs.append(offs[-1] + len(lab))
                meta.append((name, m["lang"], m["kind"], str(m["doc"])))
            f = out / f"{split}_{k // shard:03d}.npz"
            np.savez(
                f,
                images=np.stack(imgs),
                labels=np.concatenate(labs).astype(np.float32),
                offsets=np.array(offs, np.int64),
                names=np.array([m[0] for m in meta]),
                lang=np.array([m[1] for m in meta]),
                kind=np.array([m[2] for m in meta]),
                doc=np.array([m[3] for m in meta]),
            )
            index["shards"].append({"file": f.name, "split": split, "images": len(imgs), "labels": int(offs[-1])})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def load(path):
    """dict of one shard: images, labels per image as a list of arrays, names, lang, kind, doc."""
    z = np.load(path)
    o = z["offsets"]
    return dict(
        images=z["images"],
        labels=[z["labels"][o[i] : o[i + 1]] for i in range(len(o) - 1)],
        names=z["names"],
        lang=z["lang"],
        kind=z["kind"],
        doc=z["doc"],
    )
