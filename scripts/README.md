# scripts/dataset

Drivers that build, check and release detection sets (need `pip install -e ".[data]"` and the corpus; data roots from
`MULTILANG_ROOT`, see `dspages/multilang/paths.py`). Text, translation and form-filling drivers live in
[ds-page-generation `tools/multilang`](https://github.com/pavel9860/ds-page-generation/tree/main/tools/multilang).

| script | what |
|---|---|
| `build_3250.py` | the 3,250-page set with exact per-language / per-kind quotas (`hwdet.data.ds3k`) |
| `add_real_hw.py` | real handwritten pages of a boxed folder into a set in the ds3k layout |
| `text_pages_notes.py` | printed pages with reader notes |
| `check_set.py`, `box_ds.py` | read-only report of a set; labels drawn on its images |
| `pack_ds.py`, `release.py` | validate and pack into `.npz` shards; rebuild, check, pack and push a Kaggle version |
| `used_lists.py` | the sources a set used (`datasets/<name>/used_*.json`) |
