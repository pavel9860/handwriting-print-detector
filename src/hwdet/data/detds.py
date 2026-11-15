"""Detection set (YOLO: 0 handwriting, 1 print)
python -m hwdet.data.detds OUT [--per-lang 40] [--workers 12] [--limit N]"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from dspages.multilang import pages
from dspages.multilang.config import HW, Config
from dspages.multilang.paths import DATA as D

LAYOUTS, PARAS = D / "langds/det_layouts", D / "ga/paras.jsonl"
FORMS = (D / "langds/forms_work", D / "langds/tr/forms_work")
MIX = dict(notes=0.4, form=0.2, hand=0.3, print=0.1)
REUSE = 3
FORM_REUSE = 2
BLANK_FORMS = 2
PAPER = {"blank": 0.28, "lined": 0.20, "squared": 0.12, "margin": 0.08, "spiral": 0.08, "seyes": 0.08, "dotted": 0.08, "legal": 0.08}
MARGIN = 0.05
SKEW = ((0.7, 0.0, 0.4), (0.2, 1.0, 3.0), (0.1, 3.0, 8.0))
VAL_EVERY = 10


class Reject(Exception):
    """A page that cannot be used, with the reason (counted by the builders)."""


def is_val(doc):
    return int(hashlib.md5(doc.encode()).hexdigest(), 16) % VAL_EVERY == 0


def look(gray, rng, cfg: Config, on_paper):
    """Paper under a white page (p_paper), the scan chain (same p for every page), grey."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else gray
    if not on_paper and rng.random() < cfg.p_paper:
        from hwsynth.page import paper
        H, W = img.shape[:2]
        sheet = np.clip(paper((H, W), np.random.default_rng(rng.getrandbits(32)), None, cfg.scan_dpi, 0.0), 0, 1)
        sheet = sheet[..., None] if sheet.ndim == 2 else sheet
        img = (img.astype(np.float32) * sheet).astype(np.uint8)
    return cv2.cvtColor(pages.degrade(img, rng, cfg, cfg.p_scan_hw), cv2.COLOR_RGB2GRAY)


def rotate(img, boxes_sets, deg):
    """img and its box lists turned by deg (counter-clockwise), the corners cut off filled with the page's paper tone"""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.median(img)))
    turned = []
    for boxes in boxes_sets:
        bs = []
        for x0, y0, x1, y1 in boxes:
            c = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], float) @ M.T
            bs.append((max(c[:, 0].min(), 0), max(c[:, 1].min(), 0), min(c[:, 0].max(), W), min(c[:, 1].max(), H)))
        turned.append([b for b in bs if b[2] - b[0] > 2 and b[3] - b[1] > 2])
    return out, turned


def margins(img, boxes_sets, rng, long_side):
    """Paper of 0-MARGIN of the height added above and below (the page's own tone)"""
    H, W = img.shape[:2]
    top, bottom = int(rng.uniform(0, MARGIN) * H), int(rng.uniform(0, MARGIN) * H)
    img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=int(np.median(img)))
    k = long_side / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return img, [[(x0 * k, (y0 + top) * k, x1 * k, (y1 + top) * k) for x0, y0, x1, y1 in bs] for bs in boxes_sets]


def skew_angle(rng, table=SKEW):
    """A page skew in degrees: (share, low, high) bands of table, either sign."""
    u, acc = rng.random(), 0.0
    for share, a, b in table:
        acc += share
        if u <= acc:
            return rng.choice((-1, 1)) * rng.uniform(a, b)
    return 0.0


def diverse(items, X):
    """items in k-center order of their embeddings X"""
    if len(items) < 3 or X is None:
        return items
    n = np.linalg.norm(X, axis=1)
    has = np.flatnonzero(n > 0)
    if len(has) < 2:
        return items
    Z = X[has] / n[has, None]
    first = int(np.argmax(Z @ Z.mean(0)))
    order, d = [first], 1 - Z @ Z[first]
    d[first] = -np.inf
    for _ in range(len(has) - 1):
        i = int(np.argmax(d))
        order.append(i)
        d = np.minimum(d, 1 - Z @ Z[i])
        d[order] = -np.inf
    rest = [i for i in range(len(items)) if n[i] == 0]
    return [items[has[i]] for i in order] + [items[i] for i in rest]


def page_vectors(keys):
    """{(content id, page): SigLIP 2 vector} of the page-type embeddings of the corpus (emb_out: 1.38M pages; manifest id -> path)."""
    U = D.parent / "emb_out"
    want = {k[0] for k in keys}
    mid = {}
    for line in open(U / "manifest.jsonl"):
        r = json.loads(line)
        c = Path(r["path"]).name.split("_")[0]
        if c in want:
            mid[r["id"]] = c
    ids = np.load(U / "ids.npy")
    E = np.load(U / "embeddings.npy", mmap_mode="r")
    rows = np.flatnonzero(np.isin(ids[:, 0], np.fromiter(mid, int)))
    out = {(mid[int(ids[i, 0])], int(ids[i, 1])): np.asarray(E[i], np.float32) for i in rows}
    return {k: out[k] for k in keys if k in out}


def load_layouts():
    """The rendered layouts (detlayouts) with the script of their words."""
    from dspages.multilang.detlayouts import load
    from hwsynth.scripts import text_script
    out = []
    for r in load():
        if not r.get("script"):
            r["script"] = text_script(" ".join(w[4] for w in r["words"][:60]) or "a")
        out.append(r)
    return out


def passages():
    """{lang: [passage of ga/paras.jsonl without references]} for handwritten pages."""
    from dspages.multilang.dettext import CITE
    out = defaultdict(list)
    for line in open(PARAS):
        p = json.loads(line)
        items = [it for it in p["items"] if not CITE.search(it.get("t", "") if isinstance(it.get("t"), str) else "")]
        if items:
            out[p["lang"]].append(dict(p, items=items))
    return out


def sources():
    """{lang: {kind: [source, ...]}} of the layouts, the forms with answers and the passages."""
    from dspages.multilang.detlayouts import load
    from dspages.multilang.dettext import CITE
    src = defaultdict(lambda: defaultdict(list))
    assign = D / "langds/tr/assign.json"
    translated = {f["doc"] for fs in json.loads(assign.read_text()).values() for f in fs} if assign.exists() else set()
    flagged = D / "langds/layout_hw_flagged.json"
    skip = {k for k, v in json.loads(flagged.read_text()).items() if v != "clean"} if flagged.exists() else set()
    from hwsynth.scripts import text_script
    for r in load():
        if r["doc"] not in translated and r["id"] not in skip:
            if not r.get("script"):
                r["script"] = text_script(" ".join(w[4] for w in r["words"][:60]) or "a")
            src[r["lang"]]["layout"].append(r)
    meta, answers, verified = {}, {}, {}
    for d in FORMS:
        if (d / "meta.json").exists():
            meta.update(json.loads((d / "meta.json").read_text()))
            verified.update(json.loads((d / "verified.json").read_text()))
        for f in sorted(d.glob("out_*.json")):
            answers.update(json.loads(f.read_text()))
    clean = set((D / "langds/forms_checked_clean.txt").read_text().split())
    for fid, m in meta.items():
        a = {k: v for k, v in answers.get(fid, {}).items() if isinstance(v, str) and v.strip()}
        p = verified.get(fid)
        mp = Path(m["path"])
        if p is None or len(a) < 3 or (mp.parent.name == "filled_forms" and fid not in clean) or "handwritten" in mp.name:
            continue
        src[m["lang"]]["form"].append(dict(fid=fid, meta=m, answers=a, path=p, doc=mp.name.split("_")[0]))
    tr_src = {f"{l}/{f['doc']}.pdf": (f["doc"], int(f["page"])) for l, fs in
              (json.loads(assign.read_text()).items() if assign.exists() else []) for f in fs}

    def key(path, page):
        p = Path(path)
        return tr_src.get(f"{p.parent.name}/{p.name}", (p.name.split("_")[0], int(page)))
    keys = {(r["doc"], int(r["page"])) for v in src.values() for r in v["layout"]}
    keys |= {key(f["path"], f["meta"]["page"]) for v in src.values() for f in v["form"]}
    E = page_vectors(keys)
    zero = np.zeros(768, np.float32)
    for lang, v in src.items():
        lay = v["layout"]
        v["layout"] = diverse(lay, np.stack([E.get((r["doc"], int(r["page"])), zero) for r in lay]) if lay else None)
        frm = v["form"]
        blank = [f for f in frm if "forms_blank" in f["meta"]["path"]][:BLANK_FORMS]
        frm = [f for f in frm if "forms_blank" not in f["meta"]["path"]]
        frm = diverse(frm, np.stack([E.get(key(f["path"], f["meta"]["page"]), zero) for f in frm]) if frm else None) + blank
        v["form"] = frm
    for line in open(PARAS):
        p = json.loads(line)
        items = [it for it in p["items"] if not CITE.search(it.get("t", "") if isinstance(it.get("t"), str) else "")]
        if items:
            src[p["lang"]]["hand"].append(dict(p, items=items))
    return src


def jobs(per_lang):
    out = []
    for lang, s in sorted(sources().items()):
        rng = random.Random(lang)
        want = {k: round(per_lang * v) for k, v in MIX.items()}
        lay, frm = s["layout"], s["form"]
        cap = sum(1 if "forms_blank" in f["meta"]["path"] else FORM_REUSE for f in frm)
        n = dict(print=min(want["print"], len(lay)), form=min(want["form"], cap))
        n["notes"] = min(want["notes"], REUSE * len(lay) - n["print"])
        n["hand"] = per_lang - sum(n.values()) if s["hand"] else 0
        if not s["hand"]:
            n["notes"] = min(per_lang - n["print"] - n["form"], REUSE * len(lay) - n["print"])
        def rounds(pool, once=()):
            p, first = list(pool), True
            while True:
                if not first:
                    p = [x for x in p if x.get("fid") not in once]
                    rng.shuffle(p)  # noqa: B023
                yield from p
                first = False
        layout_seq = rounds(lay) if lay else None
        once = {f["fid"] for f in frm if "forms_blank" in f["meta"]["path"]}
        hand_rows = plan_rows(lang, max(0, n["hand"]), s["hand"]) if s["hand"] else []
        seqs = dict(notes=layout_seq, print=layout_seq, form=rounds(frm, once) if frm else None, hand=rounds(s["hand"]) if s["hand"] else None)
        used = Counter()
        for kind, k in n.items():
            for j in range(max(0, k)):
                src = next(seqs[kind])
                key = src.get("id") or src.get("fid") or src.get("doc")
                used[key] += 1
                doc = src.get("doc") or key
                if kind == "hand":
                    src = dict(hand_rows[j])
                    doc = src["doc"]
                out.append(dict(name=f"{lang}_{kind}_{j:03d}", lang=lang, kind=kind, src=src, doc=doc, variant=used[key],
                                split="val" if is_val(doc) else "train", row=hand_rows[j] if kind == "hand" else None))
    return out


def plan_rows(lang, n, texts):
    """n dspages.hand plan rows of a language made as dspages' own plan makes them (make_plan)"""
    if n <= 0 or not texts:
        return []
    from dspages.hand import paper as paper_mod
    from dspages.hand import plan as P
    for k in P.PAPER_BY:
        if k[1] != "table":
            P.PAPER_BY[k] = dict(PAPER)
    nrng = np.random.default_rng(int(hashlib.md5(lang.encode()).hexdigest(), 16) % 2**32)
    paper_mod.DRAW = False
    rows = P.draw_factors(n, {lang: 1}, {t["lang"] for t in texts if t["kind"] == "table"}, nrng)
    P.set_geometry(rows, nrng)
    kinds, w = list(PAPER), np.array(list(PAPER.values()), float)
    for r in rows:
        if not r.get("table"):
            r["paper"] = str(nrng.choice(kinds, p=w / w.sum()))
    for r in rows:
        if not r.get("table"):
            f = float(nrng.uniform(0.9, 1.0))
            r["_cap"] *= f / max(r["fill"], 0.05)
            r["fill"] = f
            u = float(nrng.uniform(0.3, 1.0))
            r["wj"] = [round(2.0 * u, 2), round(0.07 * u, 3), round(0.05 * u, 3), round(3.0 * u, 2)]
    caps = [r.get("_cap", 0) for r in rows]
    P.assign_texts(rows, texts, {lang: 1}, nrng)
    pool = [it for t in texts if t["kind"] != "table" for it in t["items"] if it.get("k") == "p" and it.get("t")]
    for i, r in enumerate(rows):
        if r.get("table") or not pool:
            continue
        have = sum(len(it.get("t", "")) for it in r["items"])
        k = (i * 7) % len(pool)
        while have < 0.95 * caps[i] and k < len(pool) * 2:
            it = dict(pool[k % len(pool)])
            r["items"] = list(r["items"]) + [it]
            have += len(it["t"])
            k += 1
    paper_mod.DRAW = True
    return rows


def hand_page(job, rng, cfg):
    from dspages.hand import page
    from dspages.multilang.handpages import line_boxes, reject_reason
    r = dict(job["row"], seed=int(job["row"]["seed"]) + rng.randrange(1000))
    rgb, meta = page.make(r, r["items"])
    why = reject_reason(rgb, meta, cfg)
    if why:
        raise Reject(why)
    words = [tuple(w[1]) for w in meta["words"] if w[1][2] - w[1][0] > cfg.min_box_px and w[1][3] - w[1][1] > cfg.min_box_px]
    k = cfg.long_side / max(rgb.shape[:2])
    gray = cv2.cvtColor(cv2.resize(rgb, (round(rgb.shape[1] * k), round(rgb.shape[0] * k)), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return gray, [tuple(v * k for v in b) for b in line_boxes(words)], [], True


def layout_page(job, rng, cfg):
    from dspages.multilang.notes import annotate
    r = job["src"]
    gray = cv2.imread(str(LAYOUTS / r["file"]), 0)
    if job["kind"] == "print":
        return gray, [], r["prints"], False
    from dspages.hand.plan import EXPECT
    script = EXPECT.get(job["lang"], r["script"])
    same = (r["script"] == script or {r["script"], script} <= {"Han", "Hiragana", "Katakana"}
            or job["lang"] in ("sr", "uz", "tg", "az", "ku", "mn", "kk", "ky"))
    words = r["words"] if same and r.get("text_ok", True) else []
    img, hw = annotate(gray, r["prints"], words, job["lang"], script, cfg, rng)
    if not hw:
        raise Reject("no handwriting placed")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), hw, r["prints"], False


def form_page(job, rng, cfg):
    from dspages.multilang.filledforms import fill_window
    f = job["src"]
    res = fill_window(f["fid"], f["meta"], f["answers"], job["variant"], f["path"])
    if res is None:
        return None
    gray, hw, pr, _, _ = res
    return gray, hw, pr, False


def make(job):
    cfg = Config()
    cv2.setNumThreads(1)
    d = Path(job["out"]) / job["split"]
    if (d / "labels" / f"{job['name']}.txt").exists():
        return job["name"], "done"
    fn = dict(hand=hand_page, form=form_page).get(job["kind"], layout_page)
    res = None
    for t in range(4 if job["kind"] in ("hand", "notes") else 1):
        rng = random.Random(f"{job['name']}#{t}")
        try:
            res = fn(job, rng, cfg)
        except Exception as e:
            return job["name"], f"err {e!r}"[:200]
        if res is not None:
            break
    if res is None:
        return job["name"], "rejected"
    gray, hw, prints, on_paper = res
    gray, (hw, prints) = margins(gray, [hw, prints], rng, cfg.long_side)
    gray, (hw, prints) = rotate(gray, [hw, prints], skew_angle(rng))
    img = look(gray, rng, cfg, on_paper)
    H, W = img.shape[:2]
    rows = pages.yolo_rows([(HW, hw), (1, prints)], W, H, cfg)
    for sub in ("images", "labels", "meta"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{job['name']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*cfg.jpeg_quality)])
    (d / "meta" / f"{job['name']}.json").write_text(json.dumps(dict(lang=job["lang"], kind=job["kind"], doc=job["doc"], hw=len(hw),
                                                                     print=len(prints))))
    (d / "labels" / f"{job['name']}.txt").write_text("\n".join(rows) + "\n")
    return job["name"], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-lang", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    js = [dict(j, out=a.out) for j in jobs(a.per_lang)]
    print(len(js), "pages planned:", dict(Counter(j["kind"] for j in js)), dict(Counter(j["split"] for j in js)), flush=True)
    if a.limit:
        js = random.Random(0).sample(js, a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n")
    st = Counter()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        for i, (name, s) in enumerate(ex.map(make, js, chunksize=2)):
            st[s.split()[0]] += 1
            if s.startswith("err") and st["err"] <= 10:
                print(name, s, flush=True)
            if i % 200 == 0:
                print(i, len(js), dict(st), flush=True)
    print(dict(st))


if __name__ == "__main__":
    main()
