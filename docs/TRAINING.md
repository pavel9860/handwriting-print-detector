# Training

```bash
uv pip install -e ".[train]"
hwdet data validate data/ds3k-3250-v3           # labels in 0..1, classes, sizes
hwdet data manifest data/ds3k-3250-v3           # dataset version (sha256)
hwdet train configs/train/hwdet-s.yaml --device 0,1
```
The run writes `runs/hwdet-s/` (Ultralytics outputs), `metrics.json` (our evaluation of `best.pt`) and `run.json`:
git commit and dirty flag, package versions, the full recipe, dataset name / version / sha256, metrics, sha256 of the
weights. With `MLFLOW_TRACKING_URI` set, the same record is logged to MLflow (`pip install mlflow`).

## Recipe of hwdet-s@1.0.0 (`configs/train/hwdet-s.yaml`)
| parameter | value | why |
|---|---|---|
| init | `yolo26s.pt` (COCO) | Ultralytics pretrained weights |
| optimizer | AdamW, lr0 5e-4, cosine to 5e-6, warmup 1 epoch | `optimizer=auto` would switch optimisers by iteration count; fixed for reproducibility |
| epochs | 30, patience 10 | in earlier rounds val mAP50 gained under 0.02 after epoch 8–20 |
| imgsz / batch | 1024 / 16 (nbs 16) | handwriting lines are 10–20 px high at 1024 px; smaller inputs lose them |
| mosaic | 0.5, off for the last 3 epochs | a 2 x 2 mosaic halves line height |
| scale / translate / degrees | 0.5 / 0.05 / 2 | real scans vary in size and are slightly rotated |
| fliplr | 0 | mirrored text is not text (and RTL scripts exist) |
| hsv | h 0.01, s 0.5, v 0.3 | paper tints and ink colours vary, the class does not depend on hue |
| max_det | 2400 | dense printed pages hold over 1,000 lines |

Hardware: Kaggle 2 x T4, 73 min, seed 0 (`deterministic` off for speed, so reruns differ slightly).

## Model size and CPU speed (i7-8850H, 6 threads, ONNX FP32, 1024 x 1024)
| model | params | ms / page |
|---|---:|---:|
| yolo26n | 2.6 M | 86 |
| **yolo26s** | 10.0 M | 277 |
| yolo26m | 21.9 M | 755 |

YOLO26m trained with the same recipe reached the same accuracy (val mAP50 0.931 vs 0.934; 138 vs 73 min of training, 2.7x CPU time per page), so s is shipped.
Static INT8 quantisation (ONNX QDQ) collapsed detections on this CPU without VNNI; FP16 on GPU is lossless.

## Releasing a model
1. Train, then `hwdet eval runs/hwdet-s/weights/best.pt DS --out models/metrics/<name>-<version>.json`.
2. Rename the weights `<name>-<version>.pt`, `hwdet export` for ONNX, add an entry to `models/registry.json`
   (sha256, url, dataset version, headline metrics), update `CHANGELOG.md` and `MODEL_CARD.md`.
3. `gh release create v<version> <name>-<version>.pt <name>-<version>.onnx` and tag the code.
