"""Inference: Detector("hwdet-s").predict(image) -> PageResult (text-line boxes, handwriting yes/no)."""
from dataclasses import asdict, dataclass

import numpy as np

from . import CLASSES, HW, registry


@dataclass(frozen=True)
class Box:
    cls: int
    label: str
    conf: float
    xyxy: tuple


@dataclass(frozen=True)
class PageResult:
    boxes: list
    width: int
    height: int

    @property
    def handwriting(self):
        return [b for b in self.boxes if b.cls == HW]

    def has_handwriting(self, min_boxes=1, min_conf=0.0):
        return sum(b.conf >= min_conf for b in self.handwriting) >= min_boxes

    def to_dict(self):
        return dict(width=self.width, height=self.height, n_handwriting=len(self.handwriting), boxes=[asdict(b) for b in self.boxes])


class Detector:
    """A registered model ('hwdet-s', 'hwdet-s@1.0.0') or a .pt/.onnx path; images as paths, PIL images or BGR arrays.
    fmt='onnx' runs the released ONNX export through onnxruntime."""

    def __init__(self, model=registry.DEFAULT, imgsz=None, conf=None, device=None, fmt="pt"):
        from ultralytics import YOLO
        self.weights, self.info = registry.resolve(model, fmt)
        self.imgsz = imgsz or (self.info.imgsz if self.info else 1024)
        self.conf = conf if conf is not None else (self.info.conf if self.info else 0.4)
        self.device = device
        self.model = YOLO(str(self.weights), task="detect")

    def predict(self, images, batch=8):
        """PageResult for one image, a list of them for a list."""
        single = not isinstance(images, (list, tuple))
        items = [images] if single else list(images)
        out = []
        for i in range(0, len(items), batch):
            for r in self.model.predict(items[i:i + batch], imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False):
                b = r.boxes
                xyxy, cls, conf = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())
                order = np.argsort(-conf)
                h, w = r.orig_shape
                out.append(PageResult([Box(int(cls[k]), CLASSES[cls[k]], round(float(conf[k]), 4), tuple(round(float(v), 1) for v in xyxy[k]))
                                       for k in order], w, h))
        return out[0] if single else out
