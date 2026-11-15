"""Model registry: lookup by name and version, sha256-checked local copies and downloads."""

import json

import pytest

from hwdet import registry


@pytest.fixture
def reg(tmp_path):
    w = tmp_path / "m-1.0.0.pt"
    w.write_bytes(b"weights")
    rows = [
        dict(
            name="m",
            version=v,
            file=f"m-{v}.pt",
            sha256=registry.sha256(w) if v == "1.0.0" else "0" * 64,
            url=w.as_uri(),
        )
        for v in ("0.9.0", "1.0.0", "1.10.0")
    ]
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(dict(schema=1, models=rows)))
    return p, w


def test_shipped_registry_is_valid():
    models = registry.load()
    assert models and all(len(m.sha256) == 64 and m.url.startswith("https://") for m in models)
    assert registry.get(registry.DEFAULT).version == max(
        (m.version for m in models if m.name == registry.DEFAULT), key=registry._key
    )


def test_latest_is_semver_not_lexicographic(reg):
    p, _ = reg
    assert registry.get("m", p).version == "1.10.0" and registry.get("m@0.9.0", p).version == "0.9.0"
    with pytest.raises(KeyError):
        registry.get("m@2.0.0", p)


def test_fetch_checks_hash_and_caches(reg, tmp_path):
    p, w = reg
    info = registry.get("m@1.0.0", p)
    got = registry.fetch(info, cache=tmp_path / "cache", local_dirs=())
    assert got.read_bytes() == b"weights" and got.parent == tmp_path / "cache"
    assert registry.fetch(info, cache=tmp_path / "cache", local_dirs=()) == got
    bad = registry.get("m@0.9.0", p)
    with pytest.raises(OSError):
        registry.fetch(bad, cache=tmp_path / "c2", local_dirs=())
    assert not list((tmp_path / "c2").glob("*"))


def test_resolve_accepts_a_path(tmp_path):
    w = tmp_path / "x.pt"
    w.write_bytes(b"")
    assert registry.resolve(str(w)) == (w, None)
