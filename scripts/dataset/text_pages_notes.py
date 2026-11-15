"""N text pages with notes over the languages of langds/det_layouts
python scripts/forms/text_pages_notes.py OUT_DIR [N] [SEED]"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from dspages.multilang.config import Config
from dspages.multilang.paths import LANGDS

from hwdet.data.detds import Reject, layout_page

D = LANGDS / "det_layouts"


def main():
    out, n, seed = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for d in ("pages", "boxed"):
        (out / d).mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for l in open(D / "layouts.jsonl"):
        r = json.loads(l)
        if r["type"] == "text":
            by[r["lang"]].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    cfg, got, notes = Config(), 0, {}
    while got < n and any(by.values()):
        for lang in sorted(by):
            if got >= n:
                break
            if not by[lang]:
                continue
            r = by[lang].pop()
            try:
                gray, hw, prints, _ = layout_page(dict(kind="notes", lang=lang, src=r), random.Random(f"{seed}{r['id']}"), cfg)
            except Reject:
                continue
            name = f"{lang}_{r['id']}.jpg"
            cv2.imwrite(str(out / "pages" / name), gray)
            box = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x0, y0, x1, y1 in prints:
                cv2.rectangle(box, (int(x0), int(y0)), (int(x1), int(y1)), (200, 80, 0), 1)
            for x0, y0, x1, y1 in hw:
                cv2.rectangle(box, (int(x0), int(y0)), (int(x1), int(y1)), (0, 160, 0), 2)
            cv2.imwrite(str(out / "boxed" / name), box)
            notes[r["id"]] = dict(lang=lang, notes=[[round(v) for v in b] for b in hw], prints=[[round(v) for v in b] for b in prints])
            got += 1
    (out / "notes.json").write_text(json.dumps(notes, ensure_ascii=False))
    k = [len(v["notes"]) for v in notes.values()]
    print(got, "pages,", len({v["lang"] for v in notes.values()}), "languages; notes per page: min", min(k), "median", sorted(k)[len(k) // 2], "max", max(k), "->", out)


if __name__ == "__main__":
    main()
