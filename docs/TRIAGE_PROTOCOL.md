# Issue triage study protocol

Written before the held-out test set was scored. Changes go in the log at the bottom.

## Question

Given the title and description of a GitHub issue, can a trained classifier tell whether it is a bug report, a feature request or a question, well enough to help someone sort a queue?

The model suggests. A person decides. It does not rank severity, pick owners, find duplicates or act on an issue.

## Data

The NLBSE 2024 issue report classification dataset: 3,000 labelled issues from five open source projects (facebook/react, tensorflow/tensorflow, microsoft/vscode, bitcoin/bitcoin, opencv/opencv), split by its authors into 1,500 for training and 1,500 for testing, balanced across the three labels. See `DATA.md` for the source, the pinned revision and the licence situation.

## Split

- The supplied test file is never used for training, tuning or choosing a model.
- Training rows that also occur in the test file (after trimming spaces) are removed from training (4 rows). Exact repeats inside training are removed (0 after the first step).
- A development set is cut from the remaining training rows: 20 percent, stratified by project and label, shuffled with a fixed seed (7). The model is fitted on the other 80 percent.
- The text vocabulary is fitted only on the 80 percent that is used for fitting.

## Input and labels

The input is the title, a blank line and the description, cut to 6,000 characters. The label is `bug`, `feature` or `question`. Nothing else is used as input: not the project name, the date, the author or any label-like metadata.

## Models compared

1. Majority class (the floor).
2. TF-IDF (words and word pairs) with logistic regression, with the regularisation strength C chosen from 0.5, 1, 3, 10 and 30 on the development set.
3. A small pretrained sentence transformer fine-tuned with SetFit (contrastive fine-tuning of the encoder, then a logistic regression head). Checkpoints: all-MiniLM-L6-v2 and all-MiniLM-L12-v2, 10 pair iterations, 1 epoch, batch size 16, seed 7.

The larger all-mpnet-base-v2 checkpoint used in the competition baseline was tried and stopped, because on the 16 GB Mac it took about 80 seconds per step, which would have been about 30 hours. It was never scored.

## Choosing the model

By macro-F1 on the development set. A SetFit model has to beat the best plain model by more than 0.02 to be chosen, because on 300 development rows a difference smaller than that is within noise. Otherwise the simpler model ships.

## Review threshold

The "needs review" threshold is the lowest confidence at which the accepted answers are still at least 90 percent correct on the development set. If no threshold reaches 90 percent, no review state is shown. The score is the model's confidence. It is not a calibrated probability and the page says so.

## Final test

The held-out test set is scored once, with the chosen model, by `triage final-test`. The command refuses to run again for the same release folder. The report records the model version and the hash of the release manifest, and the service only shows a report that belongs to the release it has loaded.

## Metrics

Macro-F1 (the competition's metric when the classes are balanced), accuracy, precision, recall and F1 per class, the confusion matrix, macro-F1 per project and the mean of those five, the most confident mistakes, the median and 95th percentile time for one issue on the named machine, and how many issues the review threshold defers.

## What this does not show

- The five test projects are the same five projects as in training. The result says nothing about a project the model has never seen. A separate held-out-project experiment would be needed for that.
- The labels come from each project's own labelling habits and contain noise. Some issues fit two labels.
- 1,200 fitting rows is small. A different seed could move macro-F1 by a point or two.
- The supplied template models in the competition repository were trained per project; this model is one pooled classifier. Numbers are not a leaderboard comparison.

## Log

- 1.0 (25 October 2026): first version, written before the test set was scored.
