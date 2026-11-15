"""The 3K detection set from langds/ds3k_plan.json (samples_review/select3k.py)
python -m hwdet.data.ds3k OUT [--workers 6] [--only KIND] [--limit N]"""

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from dspages.multilang import pages
from dspages.multilang.config import HW, PAGE, Config
from dspages.multilang.detlayouts import form_is_near_a4, layout_is_near_a4, near_a4  # noqa: F401
from dspages.multilang.paths import FILLING as FILLING_DIR

from . import detds as T
from .config import DS3K_SKEW as SKEW  # noqa: F401
from .config import MAX_SOURCE_SKEW, MIN_ASPECT, SHARES

PLAN = T.D / "langds/ds3k_plan.json"
FILLING = FILLING_DIR


TARGET = PAGE.long_side


def usable_forms():
    """{form id: meta} of the forms with at least 3 answers whose printed language is the language of the form (scripts forms3k_status), on a page near the A4 ratio."""
    if not (FILLING / "usable.json").exists():
        return {}
    meta = json.loads((FILLING / "meta.json").read_text())
    ok = json.loads((FILLING / "usable.json").read_text())
    return {f: m for f, m in meta.items() if ok.get(f) == "ok" and form_is_near_a4(m["path"], m["page"])}


def jobs(plan_file=PLAN):
    plan = json.loads(Path(plan_file).read_text())
    layouts = {r["id"]: r for r in T.load_layouts() if layout_is_near_a4(r["file"])}
    planned = {p["id"] for p in plan if "id" in p}
    taken = set()
    texts = T.passages()
    n_hand = Counter(p["lang"] for p in plan if p["kind"] == "hand")
    rows = {l: T.plan_rows(l, n, texts.get(l, [])) for l, n in n_hand.items()}
    fmeta = usable_forms()
    by_lang = {}
    for fid, m in fmeta.items():
        by_lang.setdefault(m["lang"], []).append(fid)
    layouts_of = {}
    for r in layouts.values():
        layouts_of.setdefault(r["lang"], []).append(r)
    out, k = [], Counter()
    for p in plan:
        lang, kind = p["lang"], p["kind"]
        j = k[(lang, kind)]
        k[(lang, kind)] += 1
        name = f"{lang}_{kind}_{j:03d}"
        if kind in ("notes", "print"):
            src = layouts.get(p.get("id")) or next(
                (r for r in layouts_of.get(lang, []) if r["id"] not in planned | taken), None
            )
            if src is None:
                continue
            taken.add(src["id"])
            doc = src["doc"]
        elif kind == "hand":
            if j >= len(rows[lang]):
                continue
            src = dict(rows[lang][j])
            for key in ("persona", "style", "pen"):
                if p.get(key):
                    src[key] = p[key]
            if p.get("persona"):
                src["type"] = p["persona"]
            doc = src["doc"]
        else:
            fids = by_lang.get(lang, [])
            if j >= len(fids):
                continue
            m = fmeta[fids[j]]
            src = dict(fid=fids[j], path=m["path"], page=int(m["page"]))
            doc = Path(m["path"]).stem.split("_")[0]
        out.append(
            dict(
                name=name,
                lang=lang,
                kind=kind,
                src=src,
                row=src if kind == "hand" else None,
                doc=doc,
                variant=p.get("variant", 0),
                split="val" if T.is_val(doc) else "train",
            )
        )
    return out


_FORMS = None


def form3k_page(job, rng, cfg):
    """A form of the plan filled with its answers (Reject when it has no prepared fields or answers)."""
    global _FORMS
    from dspages.multilang.filledforms import fill_window

    if _FORMS is None:
        meta = usable_forms()
        ans = {}
        for f in sorted(FILLING.glob("out_*.json")):
            ans.update(json.loads(f.read_text()))
        _FORMS = meta, ans
    meta, ans = _FORMS
    s = job["src"]
    a = {k: v for k, v in ans.get(s["fid"], {}).items() if isinstance(v, str) and v.strip()}
    if s["fid"] not in meta or len(a) < 3:
        raise T.Reject("form without prepared fields or answers")
    res = fill_window(s["fid"], meta[s["fid"]], a, job["variant"], s["path"])
    gray, hw, pr, _, _ = res
    return gray, hw, pr, False


def measure_skew(gray):
    """Degrees (counter-clockwise positive) that make the text lines of the page horizontal"""
    g = cv2.resize(gray, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    b = (g < np.percentile(g, 12)).astype(np.float32)
    h, w = b.shape
    best, ba = -1.0, 0.0
    for a in np.arange(-MAX_SOURCE_SKEW, MAX_SOURCE_SKEW + 0.01, 0.25):
        v = cv2.warpAffine(b, cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1), (w, h)).sum(axis=1).var()
        if v > best:
            best, ba = v, float(a)
    return ba


def content_box(gray, boxes, ink_too=True):
    """(x0, y0, x1, y1) of the page's content"""
    ink = (gray < np.median(gray) - 40) if ink_too else np.zeros(gray.shape, bool)
    ys, xs = np.nonzero(ink)
    H, W = gray.shape
    x0, y0, x1, y1 = (
        (np.percentile(xs, 0.1), np.percentile(ys, 0.1), np.percentile(xs, 99.9), np.percentile(ys, 99.9))
        if len(xs) > 50
        else (W, H, 0, 0)
        if boxes
        else (0, 0, W, H)
    )
    for bx in boxes:
        x0, y0, x1, y1 = min(x0, bx[0]), min(y0, bx[1]), max(x1, bx[2]), max(y1, bx[3])
    return max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)


def _squeeze(gray, sets, axis, keep, longest):
    """Removes the empty runs (no ink, no label) of rows (axis 0) or columns (axis 1) longer than `longest` px down to `keep` px."""
    ink = gray < np.median(gray) - 40
    for bs in sets:
        for b in bs:
            ink[max(int(b[1]), 0) : int(b[3]), max(int(b[0]), 0) : int(b[2])] = True
    prof = ink.mean(axis=1 - axis) > 0.002
    cut, i, n = [], 0, len(prof)
    while i < n:
        if prof[i]:
            i += 1
            continue
        j = i
        while j < n and not prof[j]:
            j += 1
        if j - i > longest and i > 0 and j < n:
            cut.append((i + keep, j))
        i = j
    if not cut:
        return gray, sets
    keepmask = np.ones(n, bool)
    for a, b in cut:
        keepmask[a:b] = False
    shift = np.cumsum(~keepmask)
    gray = gray[keepmask] if axis == 0 else gray[:, keepmask]
    moved = lambda v, lo, hi: float(v - shift[min(max(int(v), 0), n - 1)])
    out = []
    for bs in sets:
        nb = []
        for b in bs:
            nb.append(
                (b[0], moved(b[1], 0, 0), b[2], moved(b[3], 0, 0))
                if axis == 0
                else (moved(b[0], 0, 0), b[1], moved(b[2], 0, 0), b[3])
            )
        out.append(nb)
    return gray, out


def compact(gray, sets):
    """More text per page without more pages"""
    boxes = [b for bs in sets for b in bs]
    if not boxes:
        return gray, sets
    lh = float(np.median([min(b[3] - b[1], b[2] - b[0]) for b in boxes]))
    gray, sets = _squeeze(gray, sets, 0, int(0.5 * lh), int(1.5 * lh))
    return _squeeze(gray, sets, 1, int(0.5 * lh), int(4 * lh))


def frame(gray, sets, rng, ink_too=True):
    """The page straightened, cut to its content, then `fit`ted to the A4 ratio."""
    est = measure_skew(gray)
    if abs(est) >= 0.4:
        gray, sets = T.rotate(gray, sets, est)
    if ink_too:
        gray, sets = compact(gray, sets)
    portrait = gray.shape[0] >= gray.shape[1]
    x0, y0, x1, y1 = content_box(gray, [b for bs in sets for b in bs], ink_too)
    if portrait and (y1 - y0) < MIN_ASPECT * (x1 - x0):
        need = int(MIN_ASPECT * (x1 - x0)) - (y1 - y0)
        up = min(need // 2, y0)
        y0, y1 = y0 - up, min(y1 + need - up, gray.shape[0])
        y0 = max(0, y0 - max(0, (int(MIN_ASPECT * (x1 - x0)) - (y1 - y0))))
    gray = gray[y0:y1, x0:x1]
    sets = [[(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in bs] for bs in sets]
    return fit(gray, sets, rng)


def fit(gray, sets, rng):
    """`gray` cut to its content (cut to its content by the caller)"""
    h, w = gray.shape
    l, r, t, b = (int(rng.uniform(0, PAGE.margin) * d) for d in (w, w, h, h))
    W, H = w + l + r, h + t + b
    if W / H > PAGE.ratio:
        extra = round(W / PAGE.ratio) - H
        up = int(rng.uniform(0, 1) * extra)
        t, b = t + up, b + extra - up
    else:
        extra = round(H * PAGE.ratio) - W
        left = int(rng.uniform(0, 1) * extra)
        l, r = l + left, r + extra - left
    gray = cv2.copyMakeBorder(gray, t, b, l, r, cv2.BORDER_CONSTANT, value=int(np.median(gray)))
    sets = [[(c[0] + l, c[1] + t, c[2] + l, c[3] + t) for c in bs] for bs in sets]
    size = (round(PAGE.long_side * PAGE.ratio), PAGE.long_side)
    kx, ky = size[0] / gray.shape[1], size[1] / gray.shape[0]
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC if ky > 1 else cv2.INTER_AREA)
    return gray, [[(c[0] * kx, c[1] * ky, c[2] * kx, c[3] * ky) for c in bs] for bs in sets]


def attempt(job, src, rng, cfg):
    """One page of `src` (the job's source or an alternate): (image, rows) or Reject."""
    fn = dict(hand=T.hand_page, form=form3k_page, notes=T.layout_page, print=T.layout_page)[job["kind"]]
    gray, hw, prints, on_paper = fn(dict(job, src=src, doc=src.get("doc", job["doc"])), rng, cfg)
    gray, (hw, prints) = frame(gray, [hw, prints], rng, ink_too=job["kind"] != "hand")
    gray, (hw, prints) = T.rotate(gray, [hw, prints], T.skew_angle(rng, SKEW))
    img = T.look(gray, rng, cfg, on_paper)
    H, W = img.shape[:2]
    return img, pages.yolo_rows([(HW, hw), (1, prints)], W, H, cfg), len(hw), len(prints)


def make(job):
    """One page of the plan, built once from its own source"""
    from dataclasses import replace

    cfg = replace(Config(), long_side=TARGET)
    cv2.setNumThreads(1)
    d = Path(job["out"]) / job["split"]
    if (d / "labels" / f"{job['name']}.txt").exists():
        return job["name"], "done"
    rng = random.Random(f"{job['name']}#0")
    try:
        img, rows, n_hw, n_pr = attempt(job, job["src"], rng, cfg)
    except T.Reject as e:
        return job["name"], f"rejected: {e}"
    for sub in ("images", "labels", "meta"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(d / "images" / f"{job['name']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(*cfg.jpeg_quality)]
    )
    row = job.get("row") or {}
    (d / "meta" / f"{job['name']}.json").write_text(
        json.dumps(
            dict(
                lang=job["lang"],
                kind=job["kind"],
                doc=str(job["src"].get("doc", job["doc"])),
                hw=n_hw,
                print=n_pr,
                src={k: job["src"][k] for k in ("fid", "path", "page", "id", "file") if k in job["src"]},
                **{k: row.get(k) for k in ("persona", "style", "pen", "paper", "type") if k in row},
            )
        )
    )
    (d / "labels" / f"{job['name']}.txt").write_text("\n".join(rows) + "\n")
    return job["name"], "ok"


def run(js, workers):
    """Builds the pages of `js`; returns (Counter of results, Counter of the reasons for rejections)."""
    st, why = Counter(), Counter()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for i, (name, s) in enumerate(ex.map(make, js, chunksize=2)):
            st[s.split(":")[0]] += 1
            if s.startswith("rejected"):
                why[s.split(": ", 1)[1].split(" 0.")[0]] += 1
            if i % 200 == 0:
                print(i, len(js), dict(st), flush=True)
    return st, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan", default=str(PLAN))
    a = ap.parse_args()
    js = [dict(j, out=a.out) for j in jobs(a.plan) if not a.only or j["kind"] == a.only]
    if a.limit:
        js = random.Random(0).sample(js, min(a.limit, len(js)))
    print(len(js), "pages:", dict(Counter(j["kind"] for j in js)), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames:\n  0: handwriting\n  1: print\n"
    )
    st, why = run(js, a.workers)
    print(dict(st), "rejected because:", dict(why))


if __name__ == "__main__":
    main()


def targets(langs, total):
    """{language: pages}: total split as evenly as it goes."""
    return {l: total // len(langs) + (i < total % len(langs)) for i, l in enumerate(sorted(langs))}


def split(n):
    """{kind: pages} of n generated pages by SHARES (largest remainder)."""
    whole = sum(w for _, w in SHARES)
    raw = {k: n * w / whole for k, w in SHARES}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def quotas(langs, total, real):
    """{(language, kind): pages} of generated pages; real = {language: real pages}."""
    return {(l, k): n for l, t in targets(langs, total).items() for k, n in split(t - real.get(l, 0)).items()}


def make_plan(old, quota, spare=1):
    """Rows of the plan: the old rows of a (language, kind) first"""
    have = defaultdict(list)
    for r in old:
        have[(r["lang"], r["kind"])].append(r)
    rows = []
    for (lang, kind), n in sorted(quota.items()):
        mine = have[(lang, kind)]
        rows += mine[:n] + [{"kind": kind, "lang": lang} for _ in range(max(0, n - len(mine)))]
        rows += [{"kind": kind, "lang": lang, "spare": True} for _ in range(spare)]
    return rows


def counts(ds):
    """Counter of (language, kind) over the built pages of a set (real pages are not counted)."""
    c = Counter()
    for f in Path(ds).glob("*/meta/*.json"):
        m = json.loads(f.read_text())
        if m["kind"] != "real_hw":
            c[(m["lang"], m["kind"])] += 1
    return c


def trim(ds, quota):
    """Removes the pages beyond the quota of each (language, kind), the last by name first; returns {(language, kind)"""
    ds = Path(ds)
    by = defaultdict(list)
    for f in sorted(ds.glob("*/meta/*.json")):
        m = json.loads(f.read_text())
        if m["kind"] != "real_hw":
            by[(m["lang"], m["kind"])].append((f.parent.parent.name, f.stem))
    missing = {}
    for key, n in quota.items():
        pages = sorted(by.get(key, []), key=lambda x: x[1])
        for split_, name in pages[n:]:
            for sub, ext in (("images", "jpg"), ("labels", "txt"), ("meta", "json")):
                (ds / split_ / sub / f"{name}.{ext}").unlink()
        if len(pages) < n:
            missing[key] = n - len(pages)
    return missing
