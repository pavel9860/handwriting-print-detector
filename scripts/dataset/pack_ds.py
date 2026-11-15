"""Checks a detection set (hwdet.data.pack.validate) and packs it into uncompressed .npz shards with index.json
python scripts/forms/pack_ds.py SET_DIR OUT_DIR KAGGLE_ID [TITLE]"""

import json
import sys
from pathlib import Path

from hwdet.data import pack as dspack

TEXT = """# {title}

Handwriting (class 0) and print (class 1) text-line boxes on 768 x 1024 grey pages: forms filled by a writer model, text pages with notes, handwritten
pages, print pages and real handwritten pages, over {langs} languages. Splits are by source document.

Files `<split>_<NNN>.npz` (not compressed; `np.load` reads them, no pickle):
- `images` uint8 (n, 1024, 768) grey
- `labels` float32 (m, 5): class, cx, cy, w, h (normalised, YOLO)
- `offsets` int64 (n + 1): the labels of image i are `labels[offsets[i]:offsets[i + 1]]`
- `names`, `lang`, `kind` (form, notes, hand, print, real_hw), `doc` (source document): unicode arrays of length n

`index.json` lists the shards. {counts}

```python
import numpy as np
z = np.load("train_000.npz")
o = z["offsets"]; boxes_of_0 = z["labels"][o[0]:o[1]]; img0 = z["images"][0]
```
"""


def main():
    ds, out, kid = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    title = sys.argv[4] if len(sys.argv) > 4 else kid.split("/")[1]
    bad = dspack.validate(ds)
    print("problems:", {k: len(v) for k, v in bad.items()} or "none")
    if bad:
        sys.exit("the set has problems, nothing packed")
    index = dspack.pack(ds, out)
    langs = set()
    for s in index["shards"]:
        langs |= set(dspack.load(out / s["file"])["lang"])
    counts = ", ".join(
        f"{sp}: {sum(s['images'] for s in index['shards'] if s['split'] == sp)} pages" for sp in dspack.SPLITS
    )
    (out / "README.md").write_text(TEXT.format(title=title, langs=len(langs), counts=counts))
    (out / "dataset-metadata.json").write_text(
        json.dumps(dict(title=title, id=kid, licenses=[{"name": "other"}], isPrivate=True))
    )
    print(len(index["shards"]), "shards;", counts, "->", out)


if __name__ == "__main__":
    main()
