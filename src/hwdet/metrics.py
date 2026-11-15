"""Box matching and detection metrics (numpy only): precision/recall per class, FP/FN per page, page pass rate."""
import numpy as np


def iou(a, b):
    """IoU matrix (len(a), len(b)) of xyxy boxes."""
    return _overlap(a, b, "iou")


def containment(a, b):
    """Intersection over the smaller box area, (len(a), len(b))."""
    return _overlap(a, b, "min")


def _overlap(a, b, mode):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=2)
    area = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    den = np.minimum(area(a)[:, None], area(b)[None]) if mode == "min" else area(a)[:, None] + area(b)[None] - inter
    return inter / np.maximum(den, 1e-9)


def match(gt, pred, thresh, how="iou"):
    """Greedy one-to-one matching (pred in the given order, e.g. by confidence): (tp, fn, fp)."""
    ov = (iou if how == "iou" else containment)(gt, pred)
    used, tp = np.zeros(len(gt), bool), 0
    for j in range(ov.shape[1]):
        if not len(gt):
            break
        cand = np.where(~used & (ov[:, j] >= thresh))[0]
        if len(cand):
            used[cand[ov[cand, j].argmax()]] = True
            tp += 1
    return tp, len(gt) - tp, len(pred) - tp


def page_errors(gt, pred, thresh):
    """(per-gt miss flags, predictions matching no gt) by containment >= thresh, many-to-one allowed."""
    ov = containment(gt, pred)
    miss = ov.max(1) < thresh if ov.size else np.ones(len(gt), bool)
    fp = int((ov.max(0) < thresh).sum()) if ov.size else len(pred)
    return miss, fp


class Counts:
    """Accumulates tp/fn/fp per class over pages and FP/FN per page for one (threshold, matching) setting."""

    def __init__(self, classes):
        self.classes = classes
        self.tot = {c: np.zeros(3, int) for c in classes}
        self.pages = {c: [] for c in classes}

    def add(self, gt, pred, thresh, how="iou"):
        """gt, pred: {class: xyxy array}; pred sorted by confidence."""
        for c in self.classes:
            r = np.array(match(gt.get(c, ()), pred.get(c, ()), thresh, how))
            self.tot[c] += r
            self.pages[c].append(r)

    def summary(self, max_err=1):
        out = {}
        for c in self.classes:
            tp, fn, fp = map(int, self.tot[c])
            pp = np.array(self.pages[c]).reshape(-1, 3)
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out[c] = dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), tp=tp, fn=fn, fp=fp, pages=len(pp),
                          fn_per_page=float(pp[:, 1].mean()) if len(pp) else 0.0, fp_per_page=float(pp[:, 2].mean()) if len(pp) else 0.0,
                          pages_ok=float(((pp[:, 1] <= max_err) & (pp[:, 2] <= max_err)).mean()) if len(pp) else 0.0)
        return out
