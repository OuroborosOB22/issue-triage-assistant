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

## Did a bigger model or more data help?

We tried to beat the shipped model on the development slice. Every model used the same settings (10 iterations, 1 epoch, batch 16), three seeds each. The rule was written before the runs: a new model must beat the MiniLM mean by more than 0.02. The control ran on a Colab T4 and the others on a Colab L4. Results are in `reports/colab/sweep/` (made with `scripts/colab_sweep.py`).

| Model | Dev macro-F1, seeds 7, 11, 23 | Mean | Gain over MiniLM | Passed |
|---|---|---|---|---|
| MiniLM-L12 (control) | 0.8200, 0.7935, 0.7970 | 0.8035 | none | none |
| mpnet-base | 0.8173, 0.8230, 0.8301 | 0.8234 | +0.01996 | no, missed by 0.00004 |
| bge-base | 0.7837, 0.8268, 0.7975 | 0.8027 | -0.0008 | no |
| bge-large | not run, we stopped for time | none | none | none |

The same model and settings differ by up to 0.03 between seeds, so one run says very little. mpnet is also three times bigger and slower to serve. Nothing passed, so we kept MiniLM-L12.

We then trained the same MiniLM-L12 again on all 1,496 clean training issues (the shipped one used 1,196) and looked at the test file a second time. This is a second look, so it is less clean than the first, and the model has now seen the development issues.

| Test file, 1,500 issues | First look (shipped) | Second look (more data) |
|---|---|---|
| Macro-F1 | 0.7925 | 0.8063 |
| Accuracy | 0.7933 | 0.8067 |
| F1 bug, feature, question | 0.819, 0.830, 0.729 | 0.817, 0.846, 0.756 |
| F1 bitcoin, react, vscode, opencv, tensorflow | 0.718, 0.818, 0.773, 0.782, 0.870 | 0.723, 0.828, 0.789, 0.839, 0.851 |
| Share not flagged for review | 65.6% | 59.5% |
| Accuracy on those not flagged | 87.9% | 85.7% |
| Median time per issue | 11.8 ms | 11.2 ms |

The gain is +0.014. It is within the size that seed changes alone can cause, so we do not claim it as proof. The review threshold was set on the earlier held-out model, and it fits the retrained model a little worse. The shipped release and its page still use the first model. The retrained one is in `reports/refit-minilm/` (made with `triage train --refit-on-all`). The unseen-project study above was not repeated; its settings did not change.

## Honest limits

- The service has no login and no rate limit. Run it on your own machine only, and put access control in front of it before showing it to anyone else.
- The 0.792 is for projects that were in training. For projects it has never seen, the study above gives about 0.70.
- The competition's own template scores higher: 0.827 averaged over the five projects (react 0.872 and tensorflow 0.864 are its best). Ours is one model for all projects, with a smaller encoder, trained on 1,196 issues.
- The scores are the model's confidence, not true probabilities. The "needs review" rule leaves about two thirds of issues without a review flag.
- The labels come from each project's own labelling and contain some noise.

## Run it

Use `requirements-train.txt` for training and serving the shipped SetFit model (or `requirements.txt` for running the fast unit tests with synthetic sklearn fixtures):

```
python3.13 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt    # large, includes torch and setfit
./.venv-train/bin/pip install --no-deps -e .
./.venv-train/bin/triage fetch-data
./.venv-train/bin/triage train --setfit sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L12-v2 --iterations 10 --device mps
./.venv-train/bin/triage final-test                        # scores the test set once
./.venv-train/bin/triage serve                             # http://127.0.0.1:8090
```

Use `--device cuda` on an NVIDIA GPU, or leave it out for the CPU (slow). `--seed N` sets the training seed, and `--refit-on-all` trains the chosen SetFit model again on all training issues. Add `--out` and `--reports` to write somewhere other than `models/candidate` and `reports`, so a release is not overwritten. `python scripts/colab_sweep.py` repeats the model comparison with three seeds. Training the larger model took about 18 minutes on a 16 GB Apple silicon Mac. The `models/` folder is not committed, so train before you serve. `final-test` refuses to run twice, so the test set is only scored once.

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

Gurman Singh, Madhur Tuteja.
