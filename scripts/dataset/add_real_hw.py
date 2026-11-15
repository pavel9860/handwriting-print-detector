"""Adds the real handwritten pages of a boxed folder to a detection set in the ds3k layout (cut to the boxes, A4 ratio), class 0 only, split by document. python scripts/forms/add_real_hw.py OUT_DIR [BOXED_DIR] [HW_DIR]
python scripts/forms/add_real_hw.py OUT_DIR [BOXED_DIR] [HW_DIR]"""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from dspages.multilang import pages
from dspages.multilang.config import HW, Config
from dspages.multilang.paths import UVTM

from hwdet.data import detds as T
from hwdet.data.ds3k import TARGET, content_box, fit, near_a4


def main():
    out = Path(sys.argv[1])
    boxed = Path(sys.argv[2] if len(sys.argv) > 2 else UVTM / "tests/hw_pages800/boxed")
    hw = Path(sys.argv[3] if len(sys.argv) > 3 else UVTM / "tests/hw_pages800")
    meta, boxes = json.loads((hw / "meta.json").read_text()), json.loads((hw / "boxes.json").read_text())
    cfg = replace(Config(), long_side=TARGET)
    n = {}
    for f in sorted(boxed.glob("*.jpg")):
        name, m = f.stem, meta.get(f.stem)
        if m is None or name not in boxes:
            continue
        pg = pymupdf.open(UVTM / "Layouts" / m["path"])[m["page"]]
        k = TARGET / max(pg.rect.width, pg.rect.height)
        px = pg.get_pixmap(matrix=pymupdf.Matrix(k, k), alpha=False, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w).copy()
        page = cv2.imread(str(hw / f"{name}.jpg")).shape[:2]
        if not near_a4(page[1], page[0]):
            continue
        s = TARGET / max(page)
        lines = [tuple(v * s for v in b) for b in boxes[name]]
        x0, y0, x1, y1 = content_box(img, lines, ink_too=False)
        img = img[y0:y1, x0:x1]
        img, (lines,) = fit(img, [[(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in lines]], random.Random(name))
        rows = pages.yolo_rows([(HW, lines)], img.shape[1], img.shape[0], cfg)
        doc = Path(m["path"]).stem.split("_")[0]
        split = "val" if T.is_val(doc) else "train"
        for sub in ("images", "labels", "meta"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / split / "images" / f"real_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / split / "meta" / f"real_{name}.json").write_text(json.dumps(dict(lang=m["lang"], kind="real_hw", doc=doc, hw=len(rows), print=0, src=dict(path=m["path"], page=m["page"]))))
        (out / split / "labels" / f"real_{name}.txt").write_text("\n".join(rows) + "\n")
        n[split] = n.get(split, 0) + 1
    (out / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
    print("real handwritten pages added:", n, "->", out)


if __name__ == "__main__":
    main()
