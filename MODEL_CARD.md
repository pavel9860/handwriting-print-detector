# Model card: hwdet-s

| | |
|---|---|
| Versions | `hwdet-s@1.0.0` (current), `hwdet-s@0.4.0` (round 4, superseded) — `models/registry.json` |
| Task | text-line detection, classes `0 handwriting`, `1 print` |
| Architecture | YOLO26s (Ultralytics 8.4.175), 10.0 M parameters, NMS-free head, input 1024 px long side |
| Initialisation | `yolo26s.pt` (COCO) |
| Training data | ds3k-3250-v3: 2,883 train pages in 90 languages (synthetic forms, notes, handwritten pages, print-only pages, 154 real handwritten pages); [docs/DATASET.md](docs/DATASET.md) |
| Recipe | `configs/train/hwdet-s.yaml`: AdamW 5e-4 cosine, 30 epochs, batch 16, mosaic 0.5, no flips; Kaggle 2 x T4, 73 min |
| Files | `hwdet-s-1.0.0.pt` (20 MB, sha256 `955acf56…`), release assets |
| License | weights AGPL-3.0 (Ultralytics); code Apache-2.0 ([NOTICE](NOTICE)) |

## Intended use
Finding handwritten and printed lines on document images (scans, renders, photos of flat pages) to route documents
(filled vs blank form, handwritten vs typed page), to crop lines for handwriting OCR, or to flag annotations on printed
pages. Not intended for identifying writers or for decisions about people.

## Metrics (validation of ds3k-3250-v3, 366 pages, IoU 0.5, conf 0.4)
| class | precision | recall | F1 | FP / page | FN / page |
|---|---:|---:|---:|---:|---:|
| handwriting | 0.948 | 0.924 | 0.936 | 0.71 | 1.08 |
| print | 0.900 | 0.873 | 0.886 | 2.86 | 3.76 |

Handwriting on the 31 real handwritten validation pages: precision 0.889, recall 0.713. mAP50 0.934, mAP50-95 0.687.
Per kind and at other thresholds: `models/metrics/hwdet-s-1.0.0.json` (reproduce: `hwdet eval hwdet-s DS`).

`hwdet-s@0.4.0`: precision 0.926, recall 0.871, mAP50 0.916 on its own synthetic val (600 pages of the 0.4.0 build); not
comparable with the numbers above.

## Limitations and risks
- Trained mostly on synthetic handwriting; real pages, especially dense or faint writing and phone photos, have lower recall.
- Decorative typefaces and some Arabic print can be detected as handwriting; short entries in form cells as print.
- Scripts without a handwriting font in hwsynth were seen only as print.
- No labelled real filled-form test set exists yet; page-level thresholds should be tuned on the target documents.
