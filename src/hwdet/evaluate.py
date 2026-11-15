"""Evaluation on a YOLO set: precision / recall / F1 per class, FP and FN per page and the share of pages with at most one
handwriting error, at several confidence and IoU thresholds, overall and per page kind; box counts on unlabelled folders."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from . import CLASSES, HW
from .metrics import Counts

log = logging.getLogger(__name__)


def read_labels(path, w, h):
    """{class: xyxy px array} of a YOLO txt file."""
    rows = (
        np.loadtxt(path, ndmin=2).reshape(-1, 5)
        if Path(path).exists() and Path(path).stat().st_size
        else np.zeros((0, 5))
    )
    c, x, y, bw, bh = rows.T
    xyxy = np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], 1)
    return {k: xyxy[c == i] for i, k in enumerate(CLASSES)}


def page_kind(meta_file, stem):
    """Kind of a page from its meta file, else from the ds3k file name (<lang>_<kind>_NNN, real_<lang>_...)."""
    if meta_file.exists():
        return json.loads(meta_file.read_text()).get("kind", "?")
    parts = stem.split("_")
    return "real_hw" if parts[0] == "real" else parts[1] if len(parts) == 3 else "?"


def dataset(weights, ds, split="val", conf=(0.25, 0.4), iou=(0.5, 0.3), imgsz=1024, batch=8, device=None, limit=None):
    """metrics dict {conf_<c>: {iou_<t>: {all: {class: ...}, by_kind: {kind: {class: ...}}}}}."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    ds = Path(ds)
    imgs = sorted((ds / split / "images").glob("*.*"))[:limit]
    settings = [(c, t) for c in conf for t in iou]
    acc = {s: {"all": Counts(CLASSES)} for s in settings}
    for i in range(0, len(imgs), batch):
        chunk = imgs[i : i + batch]
        for f, r in zip(
            chunk, model.predict([str(p) for p in chunk], imgsz=imgsz, conf=min(conf), device=device, verbose=False)
        ):
            h, w = r.orig_shape
            gt = read_labels(ds / split / "labels" / f"{f.stem}.txt", w, h)
            kind = page_kind(ds / split / "meta" / f"{f.stem}.json", f.stem)
            b = r.boxes
            xyxy, cls, cf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
            for c, t in settings:
                keep = np.argsort(-cf)
                keep = keep[cf[keep] >= c]
                pred = {k: xyxy[keep][cls[keep] == j] for j, k in enumerate(CLASSES)}
                for key in ("all", kind):
                    acc[(c, t)].setdefault(key, Counts(CLASSES)).add(gt, pred, t)
    out = dict(weights=str(weights), split=split, pages=len(imgs))
    for (c, t), a in acc.items():
        out.setdefault(f"conf_{c}", {})[f"iou_{t}"] = dict(
            all=a["all"].summary(), by_kind={k: v.summary() for k, v in sorted(a.items()) if k != "all"}
        )
    return out


def report(m):
    """Markdown table of a metrics dict."""
    rows = [
        "| conf | IoU | class | precision | recall | F1 | FP/page | FN/page | pages ≤1 FP & ≤1 FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ck, by_iou in m.items():
        if not ck.startswith("conf_"):
            continue
        for ik, v in by_iou.items():
            for c, s in v["all"].items():
                rows.append(
                    f"| {ck[5:]} | {ik[4:]} | {c} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
                    f"{s['fp_per_page']:.2f} | {s['fn_per_page']:.2f} | {s['pages_ok']:.1%} |"
                )
    return "\n".join(rows)


def folders(weights, dirs, out_dir, n=250, imgsz=1024, conf=0.4, seed=0):
    """Unlabelled folders: handwriting boxes per page, pages written annotated as out_dir/<folder>/<hw count>hw_<name>.jpg."""
    import random

    from ultralytics import YOLO

    model, counts = YOLO(str(weights), task="detect"), {}
    for d in map(Path, dirs):
        files = sorted(
            p for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
        )
        random.Random(seed).shuffle(files)
        out = Path(out_dir) / d.name
        out.mkdir(parents=True, exist_ok=True)
        counts[d.name] = []
        for f in files[:n]:
            img = cv2.imread(str(f))
            if img is None:
                log.warning("unreadable %s", f)
                continue
            b = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0].boxes
            th = max(2, round(max(img.shape[:2]) / 500))
            for x, c in zip(b.xyxy.int().tolist(), b.cls.int().tolist()):
                cv2.rectangle(img, x[:2], x[2:], (255, 0, 255) if c == HW else (0, 170, 0), th * (2 if c == HW else 1))
            k = int((b.cls == HW).sum())
            cv2.imwrite(str(out / f"{k:03d}hw_{f.stem[:40]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            counts[d.name].append(k)
    return {
        d: dict(pages=len(c), hw_per_page=float(np.mean(c)) if c else 0.0, without_hw=int(np.sum(np.array(c) == 0)))
        for d, c in counts.items()
    }
