"""Versioned dataset manifest: content hash of every image and label, counts per split / class / kind / language.
The hash names the dataset version a model was trained on (registry `dataset`, run.json `data_sha`)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .. import CLASSES

SPLITS = ("train", "val", "test")


def build(ds, name=None):
    """dict manifest of a YOLO set ds/<split>/{images,labels[,meta]}; written to ds/manifest.json and ds/data.yaml."""
    ds = Path(ds)
    h, out = hashlib.sha256(), dict(name=name or ds.name, classes=list(CLASSES), splits={})
    for split in SPLITS:
        imgs = sorted((ds / split / "images").glob("*.*"))
        if not imgs:
            continue
        cls, kinds, langs, empty = Counter(), Counter(), Counter(), 0
        for img in imgs:
            lab = ds / split / "labels" / f"{img.stem}.txt"
            rows = lab.read_text().split() if lab.exists() else []
            h.update(f"{split}/{img.name}".encode())
            h.update(hashlib.sha256(img.read_bytes()).digest())
            h.update(lab.read_bytes() if lab.exists() else b"")
            c = Counter(rows[::5])
            cls.update({CLASSES[int(k)]: v for k, v in c.items()})
            empty += "0" not in c
            meta = ds / split / "meta" / f"{img.stem}.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                kinds[m.get("kind", "?")] += 1
                langs[m.get("lang", "?")] += 1
        out["splits"][split] = dict(
            pages=len(imgs), pages_without_handwriting=empty, boxes=dict(cls), kinds=dict(kinds), languages=len(langs)
        )
    out["sha256"] = h.hexdigest()
    out["version"] = out["sha256"][:12]
    (ds / "manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (ds / "data.yaml").write_text(
        "path: .\n"
        + "".join(f"{s}: {s}/images\n" for s in out["splits"])
        + "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    return out
