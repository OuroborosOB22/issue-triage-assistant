import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import artifacts, data, metrics, train, unseen

RAW = "data/raw/nlbse2024"
RELEASES = "models"
REPORTS = "reports"


def show(value):
    print(json.dumps(value, indent=1, sort_keys=True))


def cmd_fetch(args):
    show({"files": data.fetch(args.raw), "commit": data.COMMIT})


def cmd_inventory(args):
    prepared = train.prepare(args.raw)
    show({"fit": data.inventory(prepared["fit"]), "dev": data.inventory(prepared["dev"]), "test": data.inventory(prepared["test"]),
          "train_rows_dropped": prepared["dropped"]})


def dataset_facts(prepared):
    return {"source": "nlbse2024/issue-report-classification", "commit": data.COMMIT, "sha256": data.SHA256, "fit_rows": len(prepared["fit"]),
            "dev_rows": len(prepared["dev"]), "test_rows": len(prepared["test"]), "split_seed": data.SPLIT_SEED, "train_rows_dropped": prepared["dropped"]}


def cmd_train(args):
    prepared = train.prepare(args.raw)
    results = train.compare_baselines(prepared["fit"], prepared["dev"])
    if args.setfit:
        results.update(train.compare_setfit(prepared["fit"], prepared["dev"], args.setfit, args.iterations, args.epochs, args.device))
    chosen = train.best_name(results)
    labels = [issue.label for issue in prepared["dev"]]
    threshold = metrics.review_threshold(labels, results[chosen]["scores"])
    dev_summary = {name: {key: value for key, value in item["dev"].items() if key != "confusion_matrix"} for name, item in results.items()}
    version = "%s-%s" % (chosen, hashlib.sha256(json.dumps(dataset_facts(prepared), sort_keys=True).encode()).hexdigest()[:8])
    out = Path(args.out)
    kwargs = (out, results[chosen]["model"], chosen, version, results[chosen]["dev"], dataset_facts(prepared), threshold,
              {"review": metrics.coverage(labels, results[chosen]["scores"], threshold)})
    (artifacts.save_setfit if chosen.startswith("setfit") else artifacts.save_sklearn)(*kwargs)
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    (Path(args.reports) / "dev_comparison.json").write_text(json.dumps({"chosen": chosen, "models": dev_summary, "dataset": dataset_facts(prepared)}, indent=1,
                                                                        sort_keys=True) + "\n")
    show({"chosen": chosen, "dev_macro_f1": {name: round(item["dev"]["macro_f1"], 4) for name, item in results.items()}, "review_threshold": threshold})


def cmd_final_test(args):
    prepared = train.prepare(args.raw)
    predictor, manifest = artifacts.load(args.model)
    digest = hashlib.sha256((Path(args.model) / artifacts.MANIFEST).read_bytes()).hexdigest()
    report = train.final_test(predictor, manifest, digest, prepared["test"], args.reports)
    show({"macro_f1": report["macro_f1"], "cross_repo_mean_f1": report["cross_repo_mean_f1"], "review": report["review"]})


def cmd_unseen(args):
    show(unseen.run(args.raw, args.reports, args.models, args.device)["mean_macro_f1_on_unseen_project"])


def cmd_serve(args):
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(args.model, args.report), host=args.host, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="triage")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("fetch-data", cmd_fetch), ("inventory", cmd_inventory)):
        sub = commands.add_parser(name)
        sub.add_argument("--raw", default=RAW)
        sub.set_defaults(handler=handler)
    sub = commands.add_parser("train")
    sub.add_argument("--raw", default=RAW)
    sub.add_argument("--out", default=RELEASES + "/candidate")
    sub.add_argument("--reports", default=REPORTS)
    sub.add_argument("--setfit", nargs="*", default=[], help="sentence-transformer checkpoints to fine-tune with SetFit")
    sub.add_argument("--iterations", type=int, default=10)
    sub.add_argument("--epochs", type=int, default=1)
    sub.add_argument("--device", default=None)
    sub.set_defaults(handler=cmd_train)
    sub = commands.add_parser("final-test")
    sub.add_argument("--raw", default=RAW)
    sub.add_argument("--model", default=RELEASES + "/candidate")
    sub.add_argument("--reports", default=REPORTS)
    sub.set_defaults(handler=cmd_final_test)
    sub = commands.add_parser("unseen-projects")
    sub.add_argument("--raw", default=RAW)
    sub.add_argument("--reports", default=REPORTS)
    sub.add_argument("--models", nargs="+", default=["majority", "tfidf", "setfit"], choices=["majority", "tfidf", "setfit"])
    sub.add_argument("--device", default=None)
    sub.set_defaults(handler=cmd_unseen)
    sub = commands.add_parser("serve")
    sub.add_argument("--model", default=RELEASES + "/candidate")
    sub.add_argument("--report", default=REPORTS + "/final_test.json")
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=8090)
    sub.set_defaults(handler=cmd_serve)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, FileExistsError, artifacts.UntrustedArtifact, data.BadData) as problem:
        print("error: %s" % problem, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
