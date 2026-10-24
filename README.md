# BreakCast

Predicts how risky it is to change one Maven dependency version, using only facts known before the build. It is a small trained model with a clear question:

> Can facts known before a candidate build predict whether a version-only Maven dependency change breaks the build or tests, for projects the model has never seen?

BreakCast runs alone. It needs a saved dataset to train and nothing else to serve predictions: no other project, no cloud machine, no language-model key.

## Where things stand

| Part | State |
|---|---|
| Data pipeline for the BUMP benchmark (571 breaking updates, 153 projects) | works, run on the real data |
| Import of verified before and after runs (contract v1 evidence bundles) | works, tested on three real runs |
| Models: majority, major-jump rule, logistic regression, gradient boosting | work, tested on a labelled fixture set |
| Grouped project splits, metrics, refusal to train on thin data | work |
| Prediction service (`/health`, `/ready`, `/v1/model-info`, `POST /v1/predict`) | works, tested |
| A trained model on real data | **not available yet**: BUMP contains only failures |
| Frozen text encoder model | not started: needs release-note or API-diff text that is not collected |

BUMP has no passing upgrades. Training refuses to run until at least 30 verified passes and 30 failures exist in the training projects. Running `breakcast train` on the real data today stops with:

```
error: not enough verified two-class data to train honestly: training has 0 passes, needs at least 30; development has 0 passes and 124 failures, it needs both
```

The model in `models/fixture` is built from made-up rows for tests and demos. Its manifest says so and the service repeats it in every answer. It is not a research result.

## Install and test

```
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt -e .
./.venv/bin/python -m pytest
```

## Use

```
breakcast fetch-bump                   # BUMP metadata into data/raw/bump (needs git and network)
breakcast build-dataset                # rows and the split manifest into data/processed
breakcast build-dataset --evidence ../javaProject/demo/evidence   # add verified runs from Sentry
breakcast inventory                    # counts, and whether training is allowed
breakcast train                        # fits and compares the models, saves models/candidate
breakcast final-test                   # scores the untouched test projects, once
breakcast make-fixture-model           # tiny labelled demo model in models/fixture
breakcast serve --model models/fixture # service on 127.0.0.1:8090
breakcast mine-candidates              # list merged one-dependency bump pull requests (set GITHUB_TOKEN)
```

Ask for a prediction:

```
curl -s -X POST localhost:8090/v1/predict -H 'content-type: application/json' -d @docs/sample_request.json
```

The request and response follow `contracts/v1/prediction_request.schema.json` and `prediction_response.schema.json`. Feature schema 1.0.0 is described in `docs/STUDY_PROTOCOL.md`.

## The remaining job to get a real model

1. Set `GITHUB_TOKEN` (read only, public data) and run `breakcast mine-candidates` to list merged single-dependency bump pull requests in the BUMP projects.
2. Run each candidate change before and after with the Maven engine in the Sentry project.
3. Run `breakcast build-dataset --evidence <folder of evidence bundles>`.
4. `breakcast inventory`, then `breakcast train`.

## Folders

| Folder | Content |
|---|---|
| `src/breakcast/` | the code |
| `tests/` | the tests and their small fixtures |
| `docs/` | study protocol, implementation status, model card |
| `notebooks/` | Colab notebook that calls the same commands |
| `contracts/v1/` | copy of the shared JSON contract |
| `data/`, `models/` | made by the code, not committed |

## Team

Gurman Singh, Madhur Tuteja, Prayag.
