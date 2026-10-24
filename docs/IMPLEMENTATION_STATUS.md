# Implementation status

Checked on 25 October 2026. Real means it ran here on the real data. Pending means it was not run.

## Real

- Download of the NLBSE 2024 data from the pinned commit, with hash checks (1,500 training rows, 1,500 test rows).
- Training of eight models on the development split: majority class, five TF-IDF and logistic regression settings, SetFit on all-MiniLM-L6-v2 and all-MiniLM-L12-v2 (about 10 and 18 minutes on the Mac's GPU).
- Selection by development macro-F1 (SetFit L12, 0.810 against 0.740 for the best plain model, over the 0.02 margin set beforehand).
- One scoring of the test set: macro-F1 0.792.
- The service answered single and batch requests with the real model, and the page showed the real report (checked in a browser).
- 23 automated tests pass (`./.venv/bin/python -m pytest`). They use small made-up issues, not the real data.

## Pending or not done

- A run on Colab. The notebook exists but was not run there.
- The all-mpnet-base-v2 comparison, because of time on this machine.
- A held-out-project experiment.
- Calibration of the scores.
- Both themes were viewed on the classify page. The batch and evaluation views were viewed in dark mode only, and the batch view was tested through the API but not through the file picker.
- Serving a SetFit release needs torch. No lighter model was shipped because none won.

## Choices made without asking

- One pooled model instead of one per project, which differs from the competition's protocol, so no leaderboard claim is made.
- Raw data is not committed because the upstream licence file is empty.
- The review threshold targets 90 percent accuracy on the development set. It reached 88 percent on the test set.
- The page's "Try an example" button cycles through four hand-written issues, one of which the model gets wrong. They are shown as they are.
