"""Read-only report of a detection set
python scripts/forms/check_set.py SET_DIR"""
import collections
import glob
import json
import statistics as st
import sys

import cv2

D = sys.argv[1].rstrip("/") + "/"
kinds, sizes, hw, langs, pr = collections.Counter(), collections.Counter(), collections.defaultdict(list), set(), collections.Counter()
none = collections.Counter()
for lab in glob.glob(D + "*/labels/*.txt"):
    m = json.load(open(lab.replace("labels", "meta").replace(".txt", ".json")))
    H, W = cv2.imread(lab.replace("labels", "images").replace(".txt", ".jpg"), 0).shape
    kinds[m["kind"]] += 1
    sizes[(W, H)] += 1
    langs.add(m["lang"])
    rows = [l.split() for l in open(lab) if l.strip()]
    hw[m["kind"]] += [float(r[4]) * 100 for r in rows if r[0] == "0"]
    pr[m["kind"]] += sum(r[0] == "1" for r in rows)
    none[m["kind"]] += not any(r[0] == "0" for r in rows)
print("pages", sum(kinds.values()), dict(kinds), "| languages", len(langs), "| sizes", dict(sizes.most_common(3)))
for k, v in sorted(hw.items()):
    v = sorted(v)
    if v:
        print(f"{k:8} hw boxes {len(v):6} | height % of page: min {v[0]:.2f}  5th pct {v[len(v) // 20]:.2f}  median {st.median(v):.2f} | under 2%: {100 * sum(x < 2 for x in v) / len(v):.1f}% | pages without hw: {none[k]}")
print("print boxes", dict(pr))
