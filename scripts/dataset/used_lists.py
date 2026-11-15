"""The sources a built detection set used, from the meta of its pages
python scripts/dataset/used_lists.py DS_DIR [OUT_DIR]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ds = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "datasets" / Path(sys.argv[1]).name)
out.mkdir(exist_ok=True)
forms, docs, pages = {}, defaultdict(lambda: dict(lang=set(), kinds=set(), pages=0)), {}
for f in sorted(ds.glob("*/meta/*.json")):
    m = json.loads(f.read_text())
    src = m.get("src", {})
    pages[f"{f.parent.parent.name}/{f.stem}"] = dict(kind=m["kind"], lang=m["lang"], doc=m["doc"], **src)
    d = docs[m["doc"]]
    d["lang"].add(m["lang"])
    d["kinds"].add(m["kind"])
    d["pages"] += 1
    if m["kind"] == "form" and "fid" in src:
        forms.setdefault(src["fid"], dict(path=src["path"], page=src.get("page"), lang=m["lang"], pages=0))["pages"] += 1
write = lambda name, obj: (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=sorted))
write("used_forms.json", forms)
write("used_docs.json", {k: dict(v, lang=sorted(v["lang"]), kinds=sorted(v["kinds"])) for k, v in sorted(docs.items())})
write("used_pages.json", pages)
print(len(pages), "pages,", len(docs), "documents,", len(forms), "forms with a recorded source ->", out)
