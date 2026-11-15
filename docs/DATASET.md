# Dataset card: ds3k-3250-v3

The training set of `hwdet-s@1.0.0`. Built by `hwdet.data.ds3k` (`scripts/dataset/build_3250.py`) with
[hwsynth](https://github.com/pavel9860/realistic_handwriting_153_scripts) and
[ds-page-generation](https://github.com/pavel9860/ds-page-generation) (`dspages.multilang`, `dspages.hand`).
The images are not published: backgrounds come from a private corpus of public PDF documents. The sources each page
used are listed in [`datasets/ds3k-3250-v3/`](../datasets/ds3k-3250-v3) (content ids of documents, forms and pages).

## Composition
3,249 pages at 768 x 1024 (A4 ratio), **90 languages** (about 36 pages each), YOLO labels, 2 classes.

| kind | train | val | what |
|---|---:|---:|---|
| `form` | 739 | 113 | real printed forms (corpus PDFs and Google-translated forms for languages short of forms), fields found automatically and filled by one hwsynth writer with language-appropriate answers (names, dates, numbers, addresses from Faker and line banks) |
| `hand` | 772 | 86 | full handwritten pages from `dspages.hand`: real structured text of the language, 24 writer types, 8 paper types (blank, lined, squared, Seyès, dotted, ...) |
| `notes` | 746 | 76 | real printed pages with a reader's notes written between and beside the lines, check marks |
| `print` | 472 | 60 | real printed pages without handwriting (negatives) |
| `real_hw` | 154 | 31 | real handwritten pages of the corpus, text lines boxed from the ink (`dspages.multilang.hwlines`) and checked |
| **total** | **2,883** | **366** | |

| boxes | train | val |
|---|---:|---:|
| handwriting (class 0) | 42,877 | 5,180 |
| print (class 1) | 92,088 | 10,789 |
| pages without handwriting | 472 | 60 |

## Generation details
- Skew: 75% of pages within ±0.3°, 20% ±0.3–1°, 5% ±1–1.8°; content framed to the page with a 5% margin.
- Scan look on part of the pages (paper age, bleed-through, uneven light, noise, blur, JPEG) from the hwsynth scan model.
- Handwriting size follows the printed line pitch of the page (never enlarged); one writer per page.
- Only scripts that hwsynth writes with a real handwriting font are written by hand; languages without such a font have
  print-only pages.

## Labels
YOLO txt (`cls cx cy w h`, normalised): 0 handwriting, 1 print. One box per text line; print lines are split at leader
runs (`....`) and wide gaps (columns, table cells); invisible text layers are dropped.

## Splits
By source document: pages that share a document, form or writer are in one split. Validation holds 366 pages
(11%), including 31 real handwritten pages.

## Versioning
`hwdet data manifest DS` writes `manifest.json` with a sha256 over every image and label, the counts above and a short
`version`; models record it in `run.json` (`data.version`).

## Known limits
- 155 of 3,249 pages are real handwriting; the rest of the handwriting is synthetic.
- No labelled real **filled forms**; real scans and phone photos are under-represented.
- Print boxes of rasterised text inside pictures are missing; handwriting over print (signatures over text,
  strike-through) is not produced.
