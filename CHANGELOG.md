# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: [SemVer](https://semver.org/).
Package versions and model versions (`models/registry.json`, `name@version`) are numbered separately; a model names the
dataset version it was trained on.

## 1.0.1 (2026-11-15)
### Changed
- Page generation (`pagegen`) moved from hwsynth to ds-page-generation (`dspages.pagegen`): hwsynth now holds only
  handwriting lines, ds-page-generation imports only hwsynth. `data` extra pins both libraries at v0.2.1.

## 1.0.0 (2026-11-15): first public release
### Added
- `hwdet.Detector` / `hwdet predict`: inference API returning typed boxes and a page-level handwriting decision.
- Model registry (`models/registry.json`, `hwdet models list|pull`): semver model ids, weights as GitHub release
  assets, sha256-checked download and cache (`HWDET_CACHE`).
- Model `hwdet-s@1.0.0` (YOLO26s on ds3k-3250-v3) and `hwdet-s@0.4.0` (round 4, formerly `hwdet-r4-s.pt`).
- `hwdet train` from YAML recipes (`configs/train/`), run tracking (`run.json`, optional MLflow).
- `hwdet eval`: precision / recall / F1, FP and FN per page, page pass rate, by page kind, at several conf / IoU.
- `hwdet data manifest|validate|pack`: content-hashed dataset versions.
- Apache-2.0 license, NOTICE (AGPL-3.0 terms for the weights), contribution guide, CI on 3.11–3.13, pre-commit.
### Changed
- Generation moved to the libraries. Handwriting lines stay in
  [hwsynth](https://github.com/pavel9860/realistic_handwriting_153_scripts); everything that builds pages goes to
  [ds-page-generation](https://github.com/pavel9860/ds-page-generation) 0.2.1, which imports only hwsynth: `hwdet.pagegen` →
  `dspages.pagegen` (handwritten pages, filled forms); text, translation, forms, page composition and the language
  dataset → `dspages.multilang`, with their drivers and tests.
- Detection-set builders under `hwdet.data` (`detds`, `ds3k`, `pack`); drivers in `scripts/dataset/`.
- Weights no longer in git; data paths only from environment variables.
### Removed
- Round-3 builder commands (`hwdet text|bank|para|pages|build`), superseded by `hwdet.data.ds3k`.
- CC BY-NC 4.0 license.

## 0.6.0 (2026-10-13)
- Data lives outside the repo (`Config.data`, `HWDET_DATA`).
- Text: no random strings. 18 languages and scripts without usable PDF text, and `am`, `mr`, `dz` (their PDFs were
  Latin text), have 150 native handwriting-style texts written by Haiku (`translate/<code>.json`); a script without its own
  language key gets the union of the lines of its languages. `kn`, `or`, `dv` are removed (Latin text, no handwriting font
  for their scripts).
- Fonts: Google Fonts (OFL) added to the hwsynth manifest (`scripts/fetch_fonts.py` in hwsynth): handwriting families
  where Google Fonts has them, the most popular families for the 11 scripts that have none.
- Pages: 50% with handwriting, 45% print only, 5% pictures without text (photo items of the page classifier, filtered by
  the page detector); paper/scan look on 80% of pages with handwriting and 30% of the others; 10% of pages carry a
  second language.
- `python -m hwdet lang`: language/script classification set, DIR/{train,val}/<key>.npz (lossless grayscale 224 px
  crops of handwriting rows, writer ids; 30 train / 8 val writers per key, disjoint; real IAM rows added to `en`).
  0.5.0 val covered 45 of 101 keys.

## 0.5.0 (2026-10-12)
- Print boxes per page capped at `max_prints` = 400 (0.4.0: 32 train pages had over 1,000, up to 3,003; TAL ran out of
  memory at 446-515 objects per image on a T4; val had none above 658, so val did not show it).
- Pages with no handwriting were 6.4% of train, all of them placement failures (438 form, 161 gap). A failed form or
  gap page is redrawn on another background (`bg_attempts` 5). Negatives are deliberate, 30% of pages: 20% real page
  with print only (`blank:`, `p_blank`), 10% paper with neither print nor handwriting (`empty`, `p_empty`).
- Split by the content hash (file-name part before the first `_`): 4 PDFs had pages in both train and val under
  different names.
- Backgrounds are deduplicated by file hash in Layouts; 0.4.0 and older builds are deleted.

## 0.4.0 (2026-10-12)
- Handwriting is never enlarged: the target line height is drawn first, an item at least that tall is picked, the
  scale is min(target / line height, 1) and `scaled_alpha` asserts it (before, 19-36% of placed lines were enlarged
  by more than 1.5x at 100 dpi). Stack pitch follows the writer's size; letter scale 0.5-1.0; paragraph pages 200 dpi.
- Ticks and crosses are off by default (`Config.ticks`).
- Faster writer in hwsynth with identical output (vectorised sub-movements and minimum-jerk bands, no index-trick
  builders in hot paths, `write(image=False)` for the bank, faster scan model).
- Build 0.4.0: train 9,295 + val 600 pages; text 43 s, bank 3.8 min (11.7k lines), para 2.2 min (295), pages 4.7 min
  (val 18 s, train 266 s) on 6 workers; 11.4 min in total (0.3.0: about 19.5 min).

## 0.3.0 (2026-10-12)
- 25 scripts with handwriting fonts (hwsynth without fallback fonts): Khmer, Malayalam, Sinhala from OFL fonts;
  Georgian (Mkhedruli, Nuskhuri), Armenian (with ligature), Tibetan, Lao, Myanmar, Gurmukhi and Ge'ez traced from
  alphabet charts (`hwsynth/scripts/chart_to_font.py`); 101 pool keys (76 languages, 25 `script:` keys) plus marks.
- Text pool: only lines in the key's script that are mostly letters; legacy-font and OCR text is dropped.
- A language whose script has a handwriting font keeps its PDFs as print backgrounds even if its text layer is
  unusable (otherwise its script would be handwriting-only in the data). No backgrounds are skipped now.
- Build 0.3.0: train 9,295 + val 600 pages; text 46 s, bank 6 min (14k lines), para 3 min (295), pages 10 min on 6
  workers.

## 0.2.0 (2026-10-12)
- Backgrounds in the 10 languages without a handwriting font are not used (the scripts were print-only in the data).
- Ticks and crosses in checkboxes are handwriting.
- Whole-page look from the hwsynth scan model and hwsynth paper for full pages, replacing the ad hoc degrade.
- `bureaucracy` is not a background folder (real signatures; the round 2 model misses part of them, so it is not a filter).
- Evaluation output goes to `Layouts/test/<model>/`.
- hwsynth without fallback fonts; new handwriting fonts for Khmer, Malayalam, Sinhala; Georgian, Armenian and Tibetan
  glyph fonts traced from alphabet charts (21 scripts, 105 keys). Backgrounds in am, lo, my, pa are not used.
- Build: train 9,295 + val 600 pages; text 1 min, bank 9 min, para 4 min, pages 14 min on 6 workers.

## 0.1.0 (2026-10-12)
- Repository split out of `pages_generation/tools`; package `hwdet` with one `Config` holding every number.
- Pipeline `text -> bank -> para -> pages -> manifest` (`python -m hwdet build`), resumable per chunk and per page.
- Dataset round 3 (`config_sha` in `ds/manifest.json`):
  - only scripts hwsynth writes with a handwriting font (10 language keys that fell back to print fonts were 10.6% of
    hw boxes in the unreleased first round-3 build);
  - one writer (sometimes two) and one language per page; val writers disjoint from train writers;
  - handwriting placed in form fields (leaders, rules, right of `Label:`), in text gaps, and on full handwritten
    pages; no margin placement;
  - printed header/footer on 50% of the full handwritten pages; invisible text layers dropped;
  - photo and solid areas no longer counted as free paper.
- Model: `models/hwdet-yolo26n-r2.*` (round 2 weights, trained on dataset round 2; known weak on real pages).
- Kaggle kernel writes `train_meta.json` (parameters, versions, manifest hash, val metrics).
