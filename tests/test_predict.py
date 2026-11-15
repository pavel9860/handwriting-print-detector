"""Detector output shape with a stub model; the real weights run under -m slow."""

from types import SimpleNamespace

import numpy as np
import pytest

from hwdet import registry
from hwdet.predict import Box, Detector, PageResult


class T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def stub(n):
    boxes = SimpleNamespace(xyxy=T([[0, 0, 10, 10], [5, 5, 20, 8]]), cls=T([1, 0]), conf=T([0.5, 0.9]))
    return [SimpleNamespace(boxes=boxes, orig_shape=(100, 80)) for _ in range(n)]


def test_results_sorted_and_labelled():
    det = Detector.__new__(Detector)
    det.imgsz, det.conf, det.device = 1024, 0.4, None
    det.model = SimpleNamespace(predict=lambda items, **k: stub(len(items)))
    r = det.predict("page.jpg")
    assert isinstance(r, PageResult) and (r.width, r.height) == (80, 100)
    assert (
        [b.label for b in r.boxes] == ["handwriting", "print"]
        and r.has_handwriting()
        and not r.has_handwriting(min_conf=0.95)
    )
    assert len(det.predict(["a", "b", "c"], batch=2)) == 3
    assert r.to_dict()["n_handwriting"] == 1


def test_box_is_plain_data():
    assert Box(0, "handwriting", 0.9, (0, 0, 1, 1)).xyxy == (0, 0, 1, 1)


@pytest.mark.slow
def test_released_model_on_the_sample():
    pytest.importorskip("ultralytics")
    from pathlib import Path

    det = Detector(registry.DEFAULT)
    r = det.predict(str(Path(__file__).parents[1] / "samples" / "handwritten_page.png"))
    assert r.has_handwriting(min_boxes=3)
