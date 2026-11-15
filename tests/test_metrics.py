"""Box matching and the per-class / per-page counts."""
import numpy as np
import pytest

from hwdet.metrics import Counts, containment, iou, match, page_errors


def test_iou_and_containment():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 5], [20, 20, 30, 30]])
    assert iou(a, b)[0].tolist() == pytest.approx([0.5, 0.0])
    assert containment(a, b)[0].tolist() == pytest.approx([1.0, 0.0])
    assert iou(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_is_one_to_one():
    gt = np.array([[0, 0, 10, 10]])
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    assert match(gt, pred, 0.5) == (1, 0, 1)
    assert match(np.zeros((0, 4)), pred, 0.5) == (0, 0, 2)
    assert match(gt, np.zeros((0, 4)), 0.5) == (0, 1, 0)


def test_page_errors_by_containment():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    pred = np.array([[0, 0, 12, 10], [100, 100, 110, 110]], float)
    miss, fp = page_errors(gt, pred, 0.5)
    assert miss.tolist() == [False, True] and fp == 1


def test_counts_summary():
    c = Counts(("handwriting", "print"))
    box = np.array([[0, 0, 10, 10]])
    c.add({"handwriting": box}, {"handwriting": box}, 0.5)
    c.add({"handwriting": box}, {"handwriting": np.zeros((0, 4)), "print": box}, 0.5)
    s = c.summary()
    assert s["handwriting"]["precision"] == 1.0 and s["handwriting"]["recall"] == 0.5
    assert s["handwriting"]["pages_ok"] == 1.0 and s["print"]["fp"] == 1 and s["print"]["fp_per_page"] == 0.5


def test_page_kind_from_name(tmp_path):
    from hwdet.evaluate import page_kind
    assert page_kind(tmp_path / "x.json", "af_form_005") == "form"
    assert page_kind(tmp_path / "x.json", "real_uk_1e25a3181e24_0") == "real_hw"
