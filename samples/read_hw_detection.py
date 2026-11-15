"""Reads a shard of the per-page detections (Labels/hw_detection/dN__pages_NNNN.npz)
python samples/read_hw_detection.py SHARD.npz [CONF]"""
import sys

import numpy as np

z = np.load(sys.argv[1])
conf_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ids, off, cls, conf = z["ids"], z["offsets"], z["cls"], z["conf"] / 255
hw = [(tuple(int(v) for v in ids[i]), int(((cls[off[i]:off[i + 1]] == 0) & (conf[off[i]:off[i + 1]] >= conf_min)).sum())) for i in range(len(ids))]
hw = [h for h in hw if h[1]]
print(len(ids), "pages;", len(hw), f"with handwriting at conf >= {conf_min};", len(z["failed"]), "failed")
print("first pages with handwriting (id, boxes):", hw[:5])
