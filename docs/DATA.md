# Data statement

## Source

NLBSE'24 Tool Competition, Issue Report Classification. Repository: https://github.com/nlbse2024/issue-report-classification, pinned at commit `2927bc67eb42db8affd16eaf3e5a6d74f3063961`.

| File | Rows | SHA-256 |
|---|---|---|
| `data/issues_train.csv` | 1,500 | `18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5` |
| `data/issues_test.csv` | 1,500 | `4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85` |

Columns: repo, created_at, label, title, body. Labels: bug, feature, question, 500 each in each file. Projects: facebook/react, tensorflow/tensorflow, microsoft/vscode, bitcoin/bitcoin, opencv/opencv, 300 rows each in each file.

## Licence

The repository's `LICENSE` file is empty, so it declares no licence. The issue text itself was written by the public contributors of those projects. Because of that, this repository does not contain the data. `triage fetch-data` downloads the two files from the pinned commit and checks both hashes, and the files stay in `data/raw/`, which git ignores. If the data is reused beyond a course project, ask the dataset authors.

## Citation

Please cite the dataset and the work behind it:

- Kallis, Colavito, Al-Kaswan, Pascarella, Chaparro, Rani. The NLBSE'24 Tool Competition. NLBSE 2024.
- Kallis, Di Sorbo, Canfora, Panichella. Predicting issue types on GitHub. Science of Computer Programming 205 (2021).
- Kallis, Di Sorbo, Canfora, Panichella. Ticket Tagger: Machine Learning Driven Issue Classification. ICSME 2019.
- Colavito, Lanubile, Novielli. Few-Shot Learning for Issue Report Classification. NLBSE 2023.

The SetFit approach follows the competition's baseline notebook and the SetFit library (https://huggingface.co/docs/setfit/index).

## What was checked

| Check | Result |
|---|---|
| Empty titles | none in either file |
| Empty descriptions | none in training, 2 in test |
| Exact repeated rows in training | none after removing rows also found in the test file |
| Training rows that also appear in the test file (compared after trimming spaces) | 4, removed from training |
| Unknown labels | none |

## Preprocessing

Title and description are joined with a blank line and cut to 6,000 characters. Nothing else is changed. No lower-casing or stripping of code, so the model sees stack traces and code blocks as they are.
