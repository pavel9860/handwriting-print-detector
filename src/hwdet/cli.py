"""hwdet command line.
hwdet predict IMAGE... [--model hwdet-s] [--conf 0.4] [--json]
hwdet models {list,pull NAME[@VERSION]}
hwdet data {manifest DS, validate DS, pack DS OUT}
hwdet train configs/train/hwdet-s.yaml [--device 0,1] [--epochs N]
hwdet eval MODEL DS [--split val] [--out metrics.json]
hwdet export MODEL [--format onnx]"""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__, registry


def _predict(a):
    from .predict import Detector

    det = Detector(a.model, conf=a.conf, imgsz=a.imgsz, device=a.device, fmt=a.fmt)
    for path, r in zip(a.images, det.predict(a.images)):
        if a.json:
            print(json.dumps(dict(image=path, **r.to_dict())))
        else:
            print(f"{path}: {len(r.handwriting)} handwriting / {len(r.boxes) - len(r.handwriting)} print lines")


def _models(a):
    if a.action == "list":
        for m in registry.load():
            print(f"{m.id:22s} {m.arch:10s} data={m.dataset:16s} {m.file}")
    else:
        print(registry.fetch(registry.get(a.name)))


def _data(a):
    from .data import manifest, pack

    if a.action == "manifest":
        print(json.dumps(manifest.build(a.ds), indent=1))
    elif a.action == "validate":
        bad = pack.validate(a.ds)
        print(json.dumps(bad, indent=1) if bad else "ok")
        sys.exit(1 if bad else 0)
    else:
        print(json.dumps(pack.pack(a.ds, a.out), indent=1))


def _train(a):
    from .train import train

    over = {k: v for k, v in dict(epochs=a.epochs, batch=a.batch).items() if v is not None}
    rec = train(a.config, out=a.out, device=a.device, **over)
    print(json.dumps(rec["metrics"].get("conf_0.4", rec["metrics"]), indent=1)[:2000])


def _eval(a):
    from . import evaluate

    weights, _ = registry.resolve(a.model)
    m = evaluate.dataset(weights, a.ds, split=a.split, imgsz=a.imgsz, device=a.device, limit=a.limit)
    if a.out:
        Path(a.out).write_text(json.dumps(m, indent=1))
    print(evaluate.report(m))


def _export(a):
    from ultralytics import YOLO

    weights, info = registry.resolve(a.model)
    print(
        YOLO(str(weights)).export(
            format=a.format, imgsz=a.imgsz or (info.imgsz if info else 1024), dynamic=True, simplify=True
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hwdet", description="Handwriting vs print text-line detector")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default=registry.DEFAULT)
    p.add_argument("--conf", type=float)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device")
    p.add_argument("--json", action="store_true")
    p.add_argument("--format", dest="fmt", choices=["pt", "onnx"], default="pt")
    p.set_defaults(fn=_predict)
    p = sub.add_parser("models")
    p.add_argument("action", choices=["list", "pull"])
    p.add_argument("name", nargs="?", default=registry.DEFAULT)
    p.set_defaults(fn=_models)
    p = sub.add_parser("data")
    p.add_argument("action", choices=["manifest", "validate", "pack"])
    p.add_argument("ds")
    p.add_argument("out", nargs="?")
    p.set_defaults(fn=_data)
    p = sub.add_parser("train")
    p.add_argument("config")
    p.add_argument("--out", default="runs")
    p.add_argument("--device")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.set_defaults(fn=_train)
    p = sub.add_parser("eval")
    p.add_argument("model")
    p.add_argument("ds")
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    p.set_defaults(fn=_eval)
    p = sub.add_parser("export")
    p.add_argument("model")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int)
    p.set_defaults(fn=_export)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
