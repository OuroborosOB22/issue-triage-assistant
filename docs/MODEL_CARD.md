# Model card: issue type classifier

**What it is.** A sentence-transformer (all-MiniLM-L12-v2, 33 million parameters) fine-tuned with SetFit, followed by a logistic regression head. Input: an issue title and description. Output: one of bug, feature, question, with a confidence score for each.

**Intended use.** Suggest a label when triaging issues in open source style projects, so a person can sort a queue faster. Not intended for: deciding anything on its own, ranking severity, assigning people, or judging the people who wrote the issues.

**Trained on.** 1,196 labelled issues from facebook/react, tensorflow/tensorflow, microsoft/vscode, bitcoin/bitcoin and opencv/opencv (the NLBSE 2024 training file, minus 4 rows that also appear in the test file and 300 rows kept for comparing models). Single training run, seed 7, 10 pair iterations, 1 epoch, batch size 16, on an Apple silicon GPU, 1,097 seconds. Package versions are in `requirements-train.txt` and in the release manifest.

**Evaluated on.** The 1,500-issue test file from the same dataset, scored once. Macro-F1 0.792, accuracy 0.793. By class: bug F1 0.819 (precision 0.820, recall 0.818), feature 0.830 (0.808, 0.852), question 0.729 (0.749, 0.710). Biggest confusions: bugs and features labelled as questions (64 and 55 issues) and questions labelled as bugs or features (71 and 74). By project, bitcoin is the hardest (0.718) and tensorflow the easiest (0.870).

**Known weaknesses.**
- Project-specific habits: the same words mean different things in a kernel-style project and a front-end library.
- Questions that describe a failure are often called bugs by the model, and bug reports phrased politely as questions are often called questions. Several of the most confident mistakes on the test set are of this kind, with scores above 0.99.
- It has only seen five projects. Results on other projects, other languages or non-English text are unmeasured.
- It reads at most 6,000 characters, and the sentence model sees about the first 256 tokens.
- The confidence score is not calibrated. A score of 0.99 does not mean 99 percent chance of being right.

**Review threshold.** Answers below a confidence of 0.989 are marked "needs review". On the development set this keeps 64 percent of issues at 90 percent accuracy. On the test set it keeps 66 percent at 88 percent accuracy.

**Data.** See `DATA.md`. The upstream repository declares no licence, so the data is not included here.

**Not done.** No test on unseen projects. No calibration study. No comparison with the larger all-mpnet-base-v2 encoder (it was tried and stopped as too slow on this machine). No Colab run.
