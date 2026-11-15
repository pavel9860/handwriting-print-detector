"""hwdet: handwriting vs print text-line detector for document pages."""
__version__ = "1.0.0"

CLASSES = ("handwriting", "print")
HW, PRINT = 0, 1


def __getattr__(name):
    if name in ("Detector", "Box", "PageResult"):
        from . import predict
        return getattr(predict, name)
    raise AttributeError(name)
