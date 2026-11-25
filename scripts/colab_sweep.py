import argparse
import gc
import json
import statistics
from pathlib import Path

from triage import train

RAW = "data/raw/nlbse2024"
OUT = Path("reports/colab/sweep")
SEEDS = (7, 11, 23)
CANDIDATES = (
    ("minilm_l12_base", "sentence-transformers/all-MiniLM-L12-v2", 10, 1),
    ("mpnet_base", "sentence-transformers/all-mpnet-base-v2", 10, 1),
    ("bge_base", "BAAI/bge-base-en-v1.5", 10, 1),
    ("bge_large", "BAAI/bge-large-en-v1.5", 10, 1),
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


def main(device="cuda", only=None, seeds=SEEDS):
    prepared = train.prepare(RAW)
    table = {}
    for name, checkpoint, iterations, epochs in CANDIDATES:
        if only and name not in only:
            continue
        reports = [run_one(prepared, name, checkpoint, iterations, epochs, seed, device) for seed in seeds]
        table[name] = summary(reports)
        print(name, table[name], flush=True)
    return table


def rebuild_summary():
    table = {}
    for name, _, _, _ in CANDIDATES:
        reports = [json.loads(path.read_text()) for path in sorted(OUT.glob(name + "_seed*.json"))]
        if reports:
            table[name] = summary(reports)
    (OUT / "summary.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    print(rebuild_summary() if args.summary_only else main(only=args.names or None, seeds=args.seeds))
    rebuild_summary()
