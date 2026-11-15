"""ds3k: page framing (any content becomes an A4-ratio page, boxes follow) and the language / kind quota plan."""

import json
import random

import numpy as np
import pytest

ds3k = pytest.importorskip("hwdet.data.ds3k", reason="needs the data extra")
PAGE = ds3k.PAGE


@pytest.mark.parametrize("h,w", [(1400, 700), (700, 1400), (1000, 1000), (1414, 1000), (300, 2000), (2000, 300)])
def test_any_content_becomes_an_a4_page_and_boxes_follow(h, w):
    gray = np.full((h, w), 200, np.uint8)
    gray[h // 4 : h // 2, w // 4 : w // 2] = 20
    box = (w // 4, h // 4, w // 2, h // 2)
    for seed in range(5):
        out, (boxes,) = ds3k.fit(gray, [[box]], random.Random(seed))
        assert out.shape == (PAGE.long_side, round(PAGE.long_side * PAGE.ratio))
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[0])
        inner = out[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3]
        assert inner.size and inner.mean() < 40 and out[0, 0] > 150


def test_the_frame_of_a_generated_page_has_the_ratio():
    gray = np.full((1440, 1018), 230, np.uint8)
    gray[100:900, 80:900] = 30
    out, (boxes,) = ds3k.frame(gray, [[(80, 100, 900, 900)]], random.Random(1))
    assert abs(out.shape[1] / out.shape[0] - PAGE.ratio) < 0.004 and len(boxes) == 1


@pytest.mark.parametrize(
    "w,h,ok",
    [
        (768, 1024, True),
        (1018, 1440, True),
        (890, 1000, True),
        (910, 1000, False),
        (590, 1000, False),
        (1024, 644, False),
        (1440, 1018, False),
        (1000, 1000, False),
    ],
)
def test_only_pages_near_the_page_ratio_are_sources(w, h, ok):
    assert ds3k.near_a4(w, h) is ok


def test_targets_and_split_add_up_exactly():
    langs = [f"l{i}" for i in range(90)]
    t = ds3k.targets(langs, 3250)
    assert sum(t.values()) == 3250 and set(t.values()) == {36, 37} and sum(v == 37 for v in t.values()) == 10
    for n in (24, 31, 36, 29):
        s = ds3k.split(n)
        assert sum(s.values()) == n and abs(s["form"] - s["hand"]) <= 1 and s["print"] < s["notes"]


def test_quotas_take_the_real_pages_first():
    q = ds3k.quotas(["a", "b"], 73, {"a": 11})
    assert (
        sum(v for (l, _), v in q.items() if l == "a") + 11 == 37 and sum(v for (l, _), v in q.items() if l == "b") == 36
    )


def test_plan_keeps_old_rows_adds_new_and_spares():
    old = [{"kind": "form", "lang": "a", "slot": i} for i in range(5)] + [{"kind": "hand", "lang": "a", "persona": "p"}]
    rows = ds3k.make_plan(old, {("a", "form"): 3, ("a", "hand"): 2}, spare=1)
    forms = [r for r in rows if r["kind"] == "form"]
    assert [r.get("slot") for r in forms[:3]] == [0, 1, 2] and forms[3].get("spare") and len(forms) == 4
    hands = [r for r in rows if r["kind"] == "hand"]
    assert hands[0]["persona"] == "p" and "persona" not in hands[1] and hands[2]["spare"]


def test_trim_keeps_the_first_pages_and_reports_what_is_missing(tmp_path):
    for split_, name, kind in [
        ("train", "a_form_000", "form"),
        ("train", "a_form_001", "form"),
        ("val", "a_form_002", "form"),
        ("train", "a_hand_000", "hand"),
        ("train", "real_x", "real_hw"),
    ]:
        for sub in ("images", "labels", "meta"):
            (tmp_path / split_ / sub).mkdir(parents=True, exist_ok=True)
        for sub, ext in (("images", "jpg"), ("labels", "txt")):
            (tmp_path / split_ / sub / f"{name}.{ext}").write_text("x")
        (tmp_path / split_ / "meta" / f"{name}.json").write_text(json.dumps(dict(lang="a", kind=kind)))
    missing = ds3k.trim(tmp_path, {("a", "form"): 2, ("a", "hand"): 2})
    assert missing == {("a", "hand"): 1} and ds3k.counts(tmp_path) == {("a", "form"): 2, ("a", "hand"): 1}
    assert (
        not (tmp_path / "val" / "images" / "a_form_002.jpg").exists()
        and (tmp_path / "train" / "images" / "real_x.jpg").exists()
    )
