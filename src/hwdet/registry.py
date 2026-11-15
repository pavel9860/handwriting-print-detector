"""Model registry: models/registry.json names every released model (name@version), its file, sha256, data and metrics.
Weights are GitHub release assets, fetched on first use into HWDET_CACHE (default ~/.cache/hwdet) and checked by sha256."""

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "models" / "registry.json"
CACHE = Path(os.environ.get("HWDET_CACHE", Path.home() / ".cache" / "hwdet"))
DEFAULT = "hwdet-s"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    file: str
    sha256: str
    url: str
    arch: str = ""
    imgsz: int = 1024
    conf: float = 0.4
    dataset: str = ""
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    onnx: dict = field(default_factory=dict)  # {file, sha256, url} of the ONNX export, if released

    @property
    def id(self):
        return f"{self.name}@{self.version}"


def load(path=REGISTRY):
    """[ModelInfo] of the registry, newest version of a name last."""
    rows = json.loads(Path(path).read_text())["models"]
    return [ModelInfo(**r) for r in rows]


def _key(v):
    return tuple(int(x) for x in v.split("."))


def get(spec=DEFAULT, path=REGISTRY):
    """ModelInfo of 'name' (latest) or 'name@version'."""
    name, _, version = spec.partition("@")
    rows = [m for m in load(path) if m.name == name and (not version or m.version == version)]
    if not rows:
        raise KeyError(f"{spec} not in {path}; known: {sorted({m.id for m in load(path)})}")
    return max(rows, key=lambda m: _key(m.version))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(info: ModelInfo, cache=CACHE, local_dirs=(REGISTRY.parent,), fmt="pt"):
    """Local path of the weights (fmt 'pt' or 'onnx'): a checked copy in local_dirs or the cache, else downloaded."""
    file, digest, url = (
        (info.file, info.sha256, info.url)
        if fmt == "pt"
        else (info.onnx["file"], info.onnx["sha256"], info.onnx["url"])
    )
    for d in (*local_dirs, cache):
        p = Path(d) / file
        if p.exists() and sha256(p) == digest:
            return p
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / (file + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if (got := sha256(tmp)) != digest:
        tmp.unlink()
        raise OSError(f"{file}: sha256 {got} != {digest}")
    return tmp.rename(cache / file)


def resolve(spec=DEFAULT, fmt="pt"):
    """(weights path, ModelInfo or None) for a registry spec or a path to a .pt / .onnx file."""
    if Path(spec).suffix in (".pt", ".onnx") and Path(spec).exists():
        return Path(spec), None
    info = get(spec)
    return fetch(info, fmt=fmt), info
