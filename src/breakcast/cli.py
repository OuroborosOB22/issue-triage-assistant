import argparse
import json
import os
import sys
from pathlib import Path

from . import artifacts, bump, candidates, dataset, evidence, fixture, splits, train

DATA = Path("data")
RAW_BUMP = DATA / "raw" / "bump"
ROWS = DATA / "processed" / "rows.jsonl"
SPLITS = DATA / "processed" / "splits.json"
RELEASES = Path("models")


def show(value):
    print(json.dumps(value, indent=1, sort_keys=True))


def cmd_fetch_bump(args):
    show(bump.fetch_benchmark(args.into))


def cmd_build_dataset(args):
    rows, skipped = bump.load_rows(args.bump)
    extra, rejected = [], []
    for folder in args.evidence:
        found, bad = evidence.load_evidence_rows(folder)
        extra += found
        rejected += bad
    kept = dataset.deduplicate(rows + extra)
    dataset.write_rows(kept, args.out)
    manifest = splits.extend_manifest(splits.load_manifest(args.splits), kept) if Path(args.splits).is_file() else splits.build_manifest(kept)
    splits.save_manifest(manifest, args.splits)
    show({"inventory": dataset.inventory(kept), "skipped_bump_files": skipped, "rejected_evidence": rejected, "split_digest": manifest["digest"]})


def cmd_inventory(args):
    rows = dataset.read_rows(args.rows)
    show({"inventory": dataset.inventory(rows), "enough_to_train": enough(rows, args.splits)})


def enough(rows, splits_path):
    try:
        train.check_enough(splits.split_rows(train.labelled(rows), splits.load_manifest(splits_path)))
        return "yes"
    except train.InsufficientData as problem:
        return str(problem)


def cmd_train(args):
    rows = dataset.read_rows(args.rows)
    manifest = splits.load_manifest(args.splits)
    models, report = train.train_all(rows, manifest)
    chosen = train.choose(report)
    version = "dev-" + dataset.file_digest(args.rows)[:8]
    artifacts.save(args.out, models[chosen], chosen, version, report[chosen], dataset.file_digest(args.rows), manifest["digest"], False,
                   {"all_models_dev": {name: report[name]["dev"] for name in report}})
    show({"chosen": chosen, "dev_pr_auc": {name: report[name]["dev"]["pr_auc"] for name in report}, "model_version": version})


def cmd_final_test(args):
    rows = dataset.read_rows(args.rows)
    manifest = splits.load_manifest(args.splits)
    models, report = train.train_all(rows, manifest)
    chosen = train.choose(report)
    show({"chosen_on_development_data": chosen, "test": train.final_test(models, rows, manifest, chosen)})


def cmd_make_fixture_model(args):
    from .models import build, to_frame
    rows = fixture.fixture_rows()
    dataset.write_rows(rows, args.rows)
    manifest = splits.build_manifest(rows)
    splits.save_manifest(manifest, args.splits)
    models, report = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    chosen = train.choose(report)
    artifacts.save(args.out, models[chosen], chosen, "fixture-1", report[chosen], dataset.file_digest(args.rows), manifest["digest"], True)
    show({"chosen": chosen, "note": "fixture model, not a research result", "out": str(args.out)})


def cmd_serve(args):
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


def cmd_mine(args):
    token = os.environ.get("GITHUB_TOKEN")
    rows = dataset.read_rows(args.rows)
    repos = sorted({row.project for row in rows if row.label_source == "bump_metadata"})[: args.limit]
    found = []
    for repo in repos:
        found += candidates.candidates_for(repo, lambda path: candidates.github_get(path, token))
    candidates.write_manifest(found, args.out)
    show({"repos_checked": len(repos), "candidates": len(found), "manifest": str(args.out), "authenticated": bool(token)})


def parser():
    top = argparse.ArgumentParser(prog="breakcast")
    sub = top.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fetch-bump", help="download the BUMP metadata files")
    one.add_argument("--into", default=RAW_BUMP)
    one.set_defaults(run=cmd_fetch_bump)
    two = sub.add_parser("build-dataset", help="turn BUMP metadata and evidence bundles into rows and a split manifest")
    two.add_argument("--bump", default=RAW_BUMP)
    two.add_argument("--evidence", action="append", default=[], help="folder of Sentry evidence bundles (repeatable)")
    two.add_argument("--out", default=ROWS)
    two.add_argument("--splits", default=SPLITS)
    two.set_defaults(run=cmd_build_dataset)
    three = sub.add_parser("inventory", help="count rows and say whether training is possible")
    three.add_argument("--rows", default=ROWS)
    three.add_argument("--splits", default=SPLITS)
    three.set_defaults(run=cmd_inventory)
    four = sub.add_parser("train", help="fit and compare the models, refuses when data is too thin")
    four.add_argument("--rows", default=ROWS)
    four.add_argument("--splits", default=SPLITS)
    four.add_argument("--out", default=RELEASES / "candidate")
    four.set_defaults(run=cmd_train)
    five = sub.add_parser("final-test", help="score the untouched test projects once")
    five.add_argument("--rows", default=ROWS)
    five.add_argument("--splits", default=SPLITS)
    five.set_defaults(run=cmd_final_test)
    six = sub.add_parser("make-fixture-model", help="build a labelled fixture model for tests and demos")
    six.add_argument("--rows", default=DATA / "processed" / "fixture_rows.jsonl")
    six.add_argument("--splits", default=DATA / "processed" / "fixture_splits.json")
    six.add_argument("--out", default=RELEASES / "fixture")
    six.set_defaults(run=cmd_make_fixture_model)
    seven = sub.add_parser("serve", help="start the prediction service")
    seven.add_argument("--model", default=os.environ.get("BREAKCAST_MODEL_DIR"))
    seven.add_argument("--host", default="127.0.0.1")
    seven.add_argument("--port", type=int, default=8090)
    seven.set_defaults(run=cmd_serve)
    eight = sub.add_parser("mine-candidates", help="list merged single-dependency bump pull requests of the BUMP projects (set GITHUB_TOKEN)")
    eight.add_argument("--rows", default=ROWS)
    eight.add_argument("--out", default=DATA / "processed" / "pass_candidates.jsonl")
    eight.add_argument("--limit", type=int, default=20)
    eight.set_defaults(run=cmd_mine)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.run(args)
    except (train.InsufficientData, dataset.InvalidRow, splits.SplitMismatch, artifacts.UntrustedArtifact, FileNotFoundError) as problem:
        print("error: %s" % problem, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
