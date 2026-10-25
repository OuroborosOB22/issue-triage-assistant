<p align="center"><img src="src/triage/web/static/logo.png" width="120" alt="Issue Triage Assistant logo"></p>

# Issue Triage Assistant

Reads the title and description of a GitHub issue and suggests one label: **bug**, **feature** or **question**. It works on one issue at a time or on a CSV file of up to 500 issues. It only suggests. A person decides, and the model never acts on an issue.

It runs alone. After training it needs no network and no paid language model: prediction is a small sentence-transformer plus a classifier running on your own machine.

## Results

Held-out test set of 1,500 issues (the supplied NLBSE 2024 test file, scored once):

| | |
|---|---|
| Macro-F1 | **0.792** |
| Accuracy | 0.793 |
| Per class F1 | bug 0.819, feature 0.830, question 0.729 |
| Per project macro-F1 | tensorflow 0.870, react 0.818, opencv 0.782, vscode 0.773, bitcoin 0.718 (mean 0.792) |
| Time for one issue | median 11.8 ms, 95th percentile 14.7 ms inside the process, on an Apple silicon Mac |

Models compared on a development set cut from the training data (1,196 rows to fit, 300 to compare):

| Model | Dev macro-F1 |
|---|---|
| Majority class | 0.167 |
| TF-IDF and logistic regression (best of five settings) | 0.740 |
| SetFit, all-MiniLM-L6-v2 | 0.763 |
| SetFit, all-MiniLM-L12-v2 (shipped) | 0.810 |

What this does not show:

- The test projects are the same five projects as in training. These numbers say nothing about a project the model has never seen.
- The competition's own SetFit template, trained per project with a larger encoder, reports about 0.86 to 0.87 per project. This model is lower: one pooled classifier, a smaller encoder, 1,200 fitting rows and one short training run. The numbers are not a leaderboard entry.
- The scores are the model's confidence, not calibrated probabilities. The "needs review" threshold (confidence at least 0.989) leaves 64 percent of development issues and 66 percent of test issues without review, and 90 percent and 88 percent of those were right. On test it misses the 90 percent target by about two points.
- The labels come from each project's own labelling and contain noise. In the page's own examples, one of four hand-written issues is labelled wrongly.

## Run it

```
python3.13 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt     # includes torch, large
export PYTHONPATH=src
./.venv-train/bin/python -m triage.cli fetch-data
./.venv-train/bin/python -m triage.cli inventory
./.venv-train/bin/python -m triage.cli train --setfit sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L12-v2 --iterations 10 --device mps
./.venv-train/bin/python -m triage.cli final-test            # scores the test set once
./.venv-train/bin/python -m triage.cli serve                 # http://127.0.0.1:8090 (front page), /app (the tool)
```

Use `--device cuda` on an NVIDIA GPU or leave it out for the CPU (slow). Training the 12-layer model took about 18 minutes on a 16 GB Apple silicon Mac. The `models/` folder is not committed, so train before serving. A release that is a plain scikit-learn model (if a simple model had won) can be served from the lighter `requirements.txt`.

`final-test` refuses to run a second time. To start a new study, delete `reports/final_test.json` on purpose.

## The service

| Route | What it does |
|---|---|
| `GET /health`, `GET /ready` | alive, and whether a model is loaded |
| `GET /v1/model-info` | model, version, data, development metrics |
| `GET /v1/report`, `GET /v1/comparison` | the saved test report (only for the loaded release) and the development comparison |
| `POST /v1/classify` | JSON `{"title": ..., "body": ...}`, answers label, scores, review state |
| `POST /v1/classify-batch` | a CSV body with a `title` column and optional `body`; add `?format=csv` for a download |
| `GET /` | the front page, with a Start button |
| `GET /app` | the working page: classify one issue, classify a CSV, see the evaluation |

Limits: titles 300 characters, descriptions 20,000, a CSV 1 MB and 500 rows. Malformed rows are reported per row. Cells that start with `=`, `+`, `-` or `@` are prefixed with a quote in CSV downloads so a spreadsheet will not run them. Responses carry a strict content security policy and the page loads only its own scripts and fonts.

A model is loaded only from the configured folder, and only if every file matches the hashes in its manifest (`TRIAGE_MODEL_DIR`, optionally `TRIAGE_MANIFEST_SHA256` to pin it).

## Files

```
src/triage/       data, models, metrics, training, release files, service, command line, web page
tests/            23 tests (data, splits, metrics, release checks, service, page)
docs/             TRIAGE_PROTOCOL.md (written before the test set was scored), DATA.md, MODEL_CARD.md, IMPLEMENTATION_STATUS.md
reports/          the development comparison and the one-time test report
notebooks/        triage_training.ipynb (repeats the training; not run on Colab)
```

## Credits

Data and baseline ideas: the NLBSE 2024 issue report classification competition (https://github.com/nlbse2024/issue-report-classification). See `docs/DATA.md` for the citation and the licence situation. Fine-tuning uses SetFit (https://huggingface.co/docs/setfit/index) on the sentence-transformers checkpoints all-MiniLM-L6-v2 and all-MiniLM-L12-v2. Fonts are Inter and JetBrains Mono (SIL Open Font Licence, files in `src/triage/web/static/fonts`).

The earlier version of this folder predicted the risk of Maven dependency upgrades. That work is kept in the tagged history (`checkpoint-final`) and is not part of this project.
