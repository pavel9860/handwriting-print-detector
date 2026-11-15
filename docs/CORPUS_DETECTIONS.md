# Per-page detections of the page corpus (hw_detection)

`hwdet-s@1.0.0` (`models/hwdet-s-1.0.0.pt`, fp16, 1024 x 768 input as is, conf floor 0.25) over every page of the Kaggle shards
`pages-768x1024-0..16`. Stored in `UVTM/Labels/hw_detection/`, one file per input shard: `d<dataset>__pages_<NNNN>.npz`,
130 files, 1,517,073 pages, 75.1M boxes (489k handwriting). Run time 2.6 h on 2 x T4, no page failed.

| array | shape / type | meaning |
|---|---|---|
| `ids` | (n, 2) int32 | page id, as in the input shard |
| `offsets` | (n + 1,) int64 | page i owns `cls[offsets[i]:offsets[i+1]]` |
| `cls` | (m,) uint8 | 0 handwriting, 1 print |
| `conf` | (m,) uint8 | `round(conf * 255)` |
| `failed` | (k, 2) int32 | ids of pages that could not be decoded or predicted (no boxes) |

No coordinates are stored. `index.json` lists each file and its page count.

```bash
python samples/read_hw_detection.py Labels/hw_detection/d9__pages_0093.npz 0.5
```
The detector is `hwdet-s@1.0.0`, YOLO26s fine-tuned on ds3k-3250-v3.
