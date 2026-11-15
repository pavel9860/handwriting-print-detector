"""Draws the labels of a detection set on its images (handwriting green, print blue). python scripts/forms/box_ds.py DS_DIR OUT_DIR"""

import sys
from pathlib import Path

import cv2

ds, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
n = hw = pr = 0
for split in ("train", "val"):
    for f in sorted((ds / split / "images").glob("*.jpg")) if (ds / split / "images").exists() else []:
        im = cv2.cvtColor(cv2.imread(str(f), 0), cv2.COLOR_GRAY2BGR)
        H, W = im.shape[:2]
        lab = ds / split / "labels" / f"{f.stem}.txt"
        for row in lab.read_text().split("\n") if lab.exists() else []:
            if not row.strip():
                continue
            c, cx, cy, w, h = row.split()[:5]
            x0, y0, x1, y1 = (
                int((float(cx) - float(w) / 2) * W),
                int((float(cy) - float(h) / 2) * H),
                int((float(cx) + float(w) / 2) * W),
                int((float(cy) + float(h) / 2) * H),
            )
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 160, 0) if c == "0" else (200, 80, 0), 2 if c == "0" else 1)
            hw += c == "0"
            pr += c != "0"
        cv2.imwrite(str(out / f"{split}_{f.name}"), im)
        n += 1
print(n, "pages boxed;", hw, "handwriting boxes,", pr, "print boxes ->", out)
