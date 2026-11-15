# Contributing

Thanks for helping. Bug reports, real-page failure cases (with permission to share the image) and pull requests are welcome.

## Setup
```bash
uv venv && uv pip install -e ".[dev,infer]"          # + ".[data]" for dataset building, ".[train]" for training
pre-commit install
```

## Before a pull request
```bash
ruff check .
pytest -q                     # fast, no weights, no corpus
pytest -q -m slow             # downloads hwdet-s and runs it on samples/
```

## Rules
- One topic per PR; tests for new behaviour; no notebooks or data in git.
- Numbers belong in a config (dataclass or `configs/*.yaml`), not inline.
- Generation belongs upstream: handwriting lines in [hwsynth](https://github.com/pavel9860/realistic_handwriting_153_scripts);
  pages, forms and layouts in [ds-page-generation](https://github.com/pavel9860/ds-page-generation) (`dspages.pagegen`, `dspages.multilang`).
- **Models**: a new model gets a new `version` in `models/registry.json` (semver: major = label/IO change,
  minor = retrain on new data, patch = re-export), its `run.json` and metrics file, and a CHANGELOG entry. Weights go to a GitHub release, never to git.
- **Datasets**: refer to a dataset by its manifest `version` (`hwdet data manifest DS`).

By contributing you agree that your contribution is licensed under Apache-2.0.
