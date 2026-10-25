<p align="center"><img src="src/triage/web/static/logo.png" width="120" alt="Issue Triage logo"></p>

# Issue Triage Assistant

Reads the title and description of a GitHub issue and suggests one label: **bug**, **feature** or **question**. It says how sure it is, and marks the unsure ones for a person to check. It only suggests. A person decides.

You can try one issue at a time, or upload a CSV of up to 500 issues. After training it needs no internet and no paid service: a small sentence model and a classifier run on your own machine.

## How well does it work

Scored once on the 1,500 test issues from the NLBSE 2024 data:

| | |
|---|---|
| Macro-F1 | **0.792** |
| Accuracy | 0.793 |
| F1 per label | bug 0.819, feature 0.830, question 0.729 |
| F1 per project | tensorflow 0.870, react 0.818, opencv 0.782, vscode 0.773, bitcoin 0.718 |
| Speed | about 12 ms per issue on an Apple silicon Mac |

Models we compared on a development slice (300 issues) cut from the training data:

| Model | Macro-F1 |
|---|---|
| Always guess the most common label | 0.167 |
| TF-IDF and logistic regression | 0.740 |
| SetFit, MiniLM-L6 | 0.763 |
| SetFit, MiniLM-L12 (the one we ship) | 0.810 |

## What about projects the model has never seen?

We also left each project out of training in turn, trained again, and scored that project's 300 test issues (`reports/unseen_projects.json`, made with `triage unseen-projects`). The shipped model's own result above is untouched.

| Project left out | Unseen: SetFit | Unseen: TF-IDF | Same project, seen in training (shipped model) |
|---|---|---|---|
| bitcoin | 0.660 | 0.657 | 0.718 |
| react | 0.756 | 0.591 | 0.818 |
| vscode | 0.625 | 0.522 | 0.773 |
| opencv | 0.688 | 0.610 | 0.782 |
| tensorflow | 0.756 | 0.619 | 0.870 |
| **average** | **0.697** | **0.600** | **0.792** |

So on a project it has not seen, expect about 0.70 macro-F1 rather than 0.79. The neural model still beats the simple one by about 0.10. Part of the drop is that each of these runs trained on about 957 issues instead of 1,196, so the gap is not all down to the unseen project.

## Honest limits

- The service has no login and no rate limit. Run it on your own machine only, and put access control in front of it before showing it to anyone else.
- The 0.792 is for projects that were in training. For projects it has never seen, the study above gives about 0.70.
- The competition's own template scores higher (about 0.86 to 0.87 per project). Ours is one model for all projects, with a smaller encoder, trained on 1,196 issues.
- The scores are the model's confidence, not true probabilities. The "needs review" rule leaves about two thirds of issues without a review flag.
- The labels come from each project's own labelling and contain some noise.

## Run it

```
python3.13 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt    # large, includes torch
export PYTHONPATH=src
./.venv-train/bin/python -m triage.cli fetch-data
./.venv-train/bin/python -m triage.cli train --setfit sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L12-v2 --iterations 10 --device mps
./.venv-train/bin/python -m triage.cli final-test          # scores the test set once
./.venv-train/bin/python -m triage.cli serve               # http://127.0.0.1:8090
```

Use `--device cuda` on an NVIDIA GPU, or leave it out for the CPU (slow). Training the larger model took about 18 minutes on a 16 GB Apple silicon Mac. The `models/` folder is not committed, so train before you serve. `final-test` refuses to run twice, so the test set is only scored once.

Open http://127.0.0.1:8090. The front page explains the idea, and **Start** opens the tool at `/app` (Classify, Batch, Evaluation).

## The service

| Route | What it does |
|---|---|
| `GET /health`, `GET /ready` | alive, and whether a model is loaded |
| `GET /v1/model-info` | model, version, data, development scores |
| `POST /v1/classify` | JSON `{"title": ..., "body": ...}`, returns the label, the scores and the review state |
| `POST /v1/classify-batch` | a CSV with a `title` column and an optional `body`; add `?format=csv` to download |

Limits: titles 300 characters, descriptions 20,000, a CSV 1 MB and 500 rows. Bad rows are reported one by one. In CSV downloads, cells that start with `=`, `+`, `-` or `@` get a leading quote so a spreadsheet does not run them. A model is loaded only if every file matches the hashes in its manifest.

## Tests

```
PYTHONPATH=src python -m pytest
```

## Data and credits

The issues come from the NLBSE 2024 issue report classification competition (https://github.com/nlbse2024/issue-report-classification), pinned to one commit. That repository has no licence file, so we do not include the data. `fetch-data` downloads it.

Please cite: Kallis, Colavito, Al-Kaswan, Pascarella, Chaparro, Rani. *The NLBSE'24 Tool Competition.* NLBSE 2024. Fine-tuning uses SetFit (https://huggingface.co/docs/setfit/index) on sentence-transformers models.

## Team

Gurman Singh, Madhur Tuteja, Paryag.
