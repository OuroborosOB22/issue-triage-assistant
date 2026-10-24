# Implementation status

Last checked by running the commands below. Real means it ran on real data. Fixture means it ran on made-up rows built for tests. Pending means it does not exist yet.

## Real

- Download of the BUMP metadata: 571 files, 556 rows after removing exact duplicates, 153 projects, all failures.
- Feature extraction from the real rows: 532 rows complete, 24 rows with unparsable versions listed as missing.
- Import of three real Sentry runs (one pass, one test regression, one compile failure) through the evidence importer.
- Refusal to train on the real data (0 passes).

## Fixture only

- Model training, comparison, saving and loading: run on 144 made-up rows from 24 made-up projects.
- The prediction service answers: `models/fixture` is labelled as a fixture in its manifest and in every response.

## Pending

- Passing upgrades with local verification (needs a GitHub token and the Sentry engine to re-run each change).
- Any metric on real held-out projects.
- Calibration assessment (until then no probability is reported).
- Release age and API-difference features.
- Frozen text encoder model.
- A Colab run: the notebook is written but not executed here.

## Checks

| Check | Result |
|---|---|
| `./.venv/bin/python -m pytest` | 72 passed |
| `breakcast train` on the real data | stops with exit code 2 and the missing counts |
| `breakcast serve --model models/fixture`, `/health`, `/ready`, `/v1/model-info`, `/v1/predict` | answered, contract-valid |

## Choices made without asking

- Minimum for training: 30 rows per class and 10 projects in the training partition, both classes in the development partition.
- Model files use joblib with a SHA-256 in the manifest, an optional pinned digest from `BREAKCAST_MODEL_SHA256`, and the service loads only the configured folder.
- Evidence bundles map to outcomes by finding code. Licence and vulnerability findings alone give `INCONCLUSIVE`.
