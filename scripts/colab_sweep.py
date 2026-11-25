import gc
import json
import statistics
import sys
from pathlib import Path

from triage import train

RAW = "data/raw/nlbse2024"
OUT = Path("reports/colab/sweep")
SEEDS = (7, 11, 23)
CANDIDATES = (
    ("minilm_l12_base", "sentence-transformers/all-MiniLM-L12-v2", 10, 1),
    ("minilm_l12_long", "sentence-transformers/all-MiniLM-L12-v2", 20, 2),
    ("mpnet_base", "sentence-transformers/all-mpnet-base-v2", 20, 2),
    ("bge_base", "BAAI/bge-base-en-v1.5", 20, 2),
    ("bge_large", "BAAI/bge-large-en-v1.5", 20, 2),
)


def run_one(prepared, name, checkpoint, iterations, epochs, seed, device):
    target = OUT / ("%s_seed%d.json" % (name, seed))
    if target.exists():
        return json.loads(target.read_text())
    results = train.compare_setfit(prepared["fit"], prepared["dev"], [checkpoint], iterations, epochs, device, seed)
    report = next(iter(results.values()))["dev"]
    report["name"] = name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    del results
    gc.collect()
    return report


def summary(reports):
    scores = [report["macro_f1"] for report in reports]
    return {"mean": round(statistics.mean(scores), 4), "spread": round(statistics.pstdev(scores), 4), "runs": len(scores),
            "mean_train_seconds": round(statistics.mean(report["train_seconds"] for report in reports), 1)}


def main(device="cuda", only=None):
    prepared = train.prepare(RAW)
    table = {}
    for name, checkpoint, iterations, epochs in CANDIDATES:
        if only and name not in only:
            continue
        reports = [run_one(prepared, name, checkpoint, iterations, epochs, seed, device) for seed in SEEDS]
        table[name] = summary(reports)
        print(name, table[name], flush=True)
    (OUT / "summary.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
