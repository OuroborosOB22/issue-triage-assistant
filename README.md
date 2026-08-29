# ML Project

Machine learning major project for semester 5.

## Status

Setting up. We have the folder structure and the ground rules in place. The topic is
not locked yet, so there is no model here to look at so far.

## How the folders are meant to be used

| folder | what goes in it |
| --- | --- |
| `src/` | all the real code |
| `data/raw/` | data exactly as we downloaded it, never edited by hand |
| `data/processed/` | anything our own code made from the raw data |
| `notebooks/` | exploring and trying things out |
| `models/` | trained models we saved |
| `reports/` | results, figures and the write up |
| `tests/` | small checks that the pipeline still runs |

One rule we are keeping from the start: everything in `data/processed/` and `models/`
has to be reproducible by running the code in `src/`. If we cannot regenerate it, we
do not trust it.

## Setup

```
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Team

Gurman Singh, Madhur Tuteja, Prayag.
