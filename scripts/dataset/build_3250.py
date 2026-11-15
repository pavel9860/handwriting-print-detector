"""The detection set with an exact number of pages (hwdet.data.ds3k)
python scripts/forms/build_3250.py OUT_DIR [--total 3250] [--workers 8] [--rounds 3]"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
from dspages.multilang.paths import UVTM

from hwdet.data import ds3k

HW = UVTM / "tests/hw_pages800"


def real_counts():
    meta = json.loads((HW / "meta.json").read_text())
    out = {}
    for n in (HW / "selected_247.txt").read_text().split():
        h, w = cv2.imread(str(HW / n), 0).shape
        if ds3k.near_a4(w, h):
            out[meta[n[:-4]]["lang"]] = out.get(meta[n[:-4]]["lang"], 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--total", type=int, default=3250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads(ds3k.PLAN.read_text())
    real = real_counts()
    quota = ds3k.quotas(sorted({r["lang"] for r in old}), a.total, real)
    rows = ds3k.make_plan(old, quota)
    (out / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n"
    )
    for rnd in range(1, a.rounds + 1):
        (out / "plan.json").write_text(json.dumps(rows))
        js = [dict(j, out=str(out)) for j in ds3k.jobs(out / "plan.json")]
        print(f"round {rnd}: {len(js)} jobs", flush=True)
        st, why = ds3k.run(js, a.workers)
        missing = ds3k.trim(out, quota)
        print(
            f"round {rnd}: {dict(st)}, rejected because {dict(why)}; still missing {sum(missing.values())}: {missing}",
            flush=True,
        )
        if not missing:
            break
        for (l, k), n in missing.items():
            if k != "hand":
                quota[(l, k)] -= n
                quota[(l, "hand")] += n
        rows += [{"kind": "hand", "lang": l, "spare": True} for (l, _), n in missing.items() for _ in range(n + 1)]
    subprocess.run([sys.executable, str(Path(__file__).parent / "add_real_hw.py"), str(out)], check=True)
    n = len(list(out.glob("*/images/*.jpg")))
    print("pages:", n, "of", a.total)


if __name__ == "__main__":
    main()
