import csv
import io
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import artifacts, data, metrics

MAX_JSON = 60000
MAX_CSV = 1_000_000
MAX_ROWS = 500
MAX_TITLE = 300
MAX_BODY = 20000
WEB = Path(__file__).resolve().parent / "web"
HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
}


class TooLarge(Exception):
    pass


async def read_limited(request, limit):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise TooLarge()
    received = bytearray()
    async for chunk in request.stream():
        received.extend(chunk)
        if len(received) > limit:
            raise TooLarge()
    return bytes(received)


def problem(code, message, status):
    return JSONResponse({"status": code, "message": message}, status_code=status)


def safe_cell(value):
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def answer(scores, threshold, version):
    label, score = metrics.top(scores)
    review = None if threshold is None else ("needs_review" if score < threshold else "ok")
    return {"label": label, "score": round(score, 4), "scores": {key: round(value, 4) for key, value in scores.items()}, "review": review,
            "model_version": version}


def parse_issue(item):
    if not isinstance(item, dict):
        raise ValueError("an issue must be an object with title and body")
    title, body = item.get("title"), item.get("body", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(body, str):
        raise ValueError("body must be text")
    if len(title) > MAX_TITLE or len(body) > MAX_BODY:
        raise ValueError("title is limited to %d and body to %d characters" % (MAX_TITLE, MAX_BODY))
    return data.join_text(title, body)


def create_app(model_dir=None, report_path=None, pinned_sha256=None):
    app = FastAPI(title="Issue triage", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"predictor": None, "manifest": None, "problem": None, "report": None, "comparison": None}
    directory = model_dir or os.environ.get("TRIAGE_MODEL_DIR")
    pin = pinned_sha256 or os.environ.get("TRIAGE_MANIFEST_SHA256")
    if directory:
        try:
            state["predictor"], state["manifest"] = artifacts.load(directory, pin)
        except (FileNotFoundError, artifacts.UntrustedArtifact) as failure:
            state["problem"] = str(failure)
    report_file = Path(report_path or os.environ.get("TRIAGE_REPORT") or (Path(directory) / "final_test.json" if directory else "final_test.json"))
    if report_file.is_file():
        state["report"] = json.loads(report_file.read_text())
    comparison_file = report_file.parent / "dev_comparison.json"
    if comparison_file.is_file():
        state["comparison"] = json.loads(comparison_file.read_text())

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        for name, value in HEADERS.items():
            response.headers[name] = value
        if request.url.path.startswith("/v1"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def unavailable():
        return problem("MODEL_UNAVAILABLE", state["problem"] or "no model release is configured", 503)

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if state["predictor"] is None:
            return JSONResponse({"status": "not ready", "reason": state["problem"] or "no model release is configured"}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/model-info")
    def model_info():
        if state["manifest"] is None:
            return unavailable()
        keep = ("task", "kind", "model_name", "model_version", "created_at", "dataset", "dev_metrics", "review_threshold", "calibrated")
        return {key: state["manifest"].get(key) for key in keep}

    @app.get("/v1/report")
    def report():
        if state["report"] is not None and state["manifest"] is not None and state["report"].get("model_version") != state["manifest"]["model_version"]:
            return problem("REPORT_UNAVAILABLE", "the saved test report belongs to a different model release", 404)
        if state["report"] is None:
            return problem("REPORT_UNAVAILABLE", "the held-out test has not been scored for this release", 404)
        return state["report"]

    @app.get("/v1/comparison")
    def comparison():
        if state["comparison"] is None:
            return problem("REPORT_UNAVAILABLE", "no development comparison has been saved", 404)
        return state["comparison"]

    @app.post("/v1/classify")
    async def classify(request: Request):
        try:
            raw = await read_limited(request, MAX_JSON)
        except TooLarge:
            return problem("TOO_LARGE", "request too large", 413)
        try:
            text = parse_issue(json.loads(raw))
        except ValueError as failure:
            return problem("INVALID_INPUT", str(failure)[:160], 422)
        if state["predictor"] is None:
            return unavailable()
        started = time.perf_counter()
        scores = (await run_in_threadpool(state["predictor"].scores, [text]))[0]
        result = answer(scores, state["manifest"]["review_threshold"], state["manifest"]["model_version"])
        result["inference_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    @app.post("/v1/classify-batch")
    async def classify_batch(request: Request, format: str = "json"):
        try:
            raw = await read_limited(request, MAX_CSV)
        except TooLarge:
            return problem("TOO_LARGE", "the CSV is limited to %d bytes" % MAX_CSV, 413)
        try:
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
            names = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
            if "title" not in names:
                return problem("INVALID_INPUT", "the CSV needs a title column (and optionally body)", 422)
            rows = []
            for line in reader:
                rows.append({"title": line.get(names["title"]) or "", "body": line.get(names["body"]) or "" if "body" in names else ""})
                if len(rows) > MAX_ROWS:
                    return problem("TOO_MANY_ROWS", "a batch is limited to %d issues" % MAX_ROWS, 413)
        except (UnicodeDecodeError, csv.Error):
            return problem("INVALID_INPUT", "the file is not valid UTF-8 CSV", 422)
        if state["predictor"] is None:
            return unavailable()
        texts, positions, results = [], [], [None] * len(rows)
        for index, item in enumerate(rows):
            try:
                texts.append(parse_issue(item))
                positions.append(index)
            except ValueError as failure:
                results[index] = {"row": index + 1, "title": item["title"][:MAX_TITLE], "error": str(failure)[:120]}
        scored = await run_in_threadpool(state["predictor"].scores, texts) if texts else []
        for index, scores in zip(positions, scored):
            results[index] = {"row": index + 1, "title": rows[index]["title"][:MAX_TITLE],
                              **answer(scores, state["manifest"]["review_threshold"], state["manifest"]["model_version"])}
        if format == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["row", "title", "label", "score", "review", "error"])
            for item in results:
                writer.writerow([item["row"], safe_cell(item["title"]), item.get("label", ""), item.get("score", ""), item.get("review") or "",
                                 safe_cell(item.get("error", ""))])
            return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="triage.csv"'})
        return {"count": len(results), "results": results}

    if WEB.is_dir():
        app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

        @app.get("/")
        def landing():
            return FileResponse(WEB / "landing.html")

        @app.get("/app")
        def home():
            return FileResponse(WEB / "index.html")

    return app
