"use strict";
var LABELS = ["bug", "feature", "question"];
var EXAMPLES = [
  {title: "Add dark mode to the settings page", body: "It would be great to have a dark theme option."},
  {title: "How do I configure the proxy for pip behind a firewall?", body: "I could not find this in the docs."},
  {title: "Segmentation fault when loading a large model", body: "Running model.load() on the 4GB checkpoint crashes with a segfault on Ubuntu 22.04. Stack trace attached."},
  {title: "App crashes when I open the settings page", body: "Steps: open settings, scroll down.\nError: TypeError: cannot read property 'map' of undefined"}
];
var nextExample = 0;
var lastBatch = null;
var filter = "all";

function el(tag, props, children) {
  var node = document.createElement(tag);
  Object.keys(props || {}).forEach(function (key) {
    if (key === "text") node.textContent = props[key]; else node.setAttribute(key, props[key]);
  });
  (children || []).forEach(function (child) { node.appendChild(typeof child === "string" ? document.createTextNode(child) : child); });
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
function pct(value) { return (value * 100).toFixed(1) + "%"; }
function fixed(value) { return value === null || value === undefined ? "n/a" : value.toFixed(3); }

function api(path, options) {
  return fetch(path, options).then(function (response) {
    return response.json().catch(function () { return {}; }).then(function (body) { return {ok: response.ok, status: response.status, body: body}; });
  });
}

function show(view) {
  var names = ["classify", "batch", "evaluation"];
  if (names.indexOf(view) < 0) view = "classify";
  names.forEach(function (name) { document.getElementById(name).hidden = name !== view; });
  Array.prototype.forEach.call(document.querySelectorAll("nav a"), function (link) {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  if (view === "evaluation") loadEvaluation();
}

function loadStatus() {
  var status = document.getElementById("status");
  api("/v1/model-info").then(function (result) {
    if (!result.ok) { status.textContent = "Model unavailable: " + (result.body.message || "no release is configured") + "."; return; }
    var info = result.body;
    status.textContent = "Model " + info.model_name + " (" + info.model_version + "). Dev macro-F1 " + fixed(info.dev_metrics && info.dev_metrics.macro_f1) + ". Scores are not calibrated.";
  });
}

function scoreBars(scores) {
  var wrap = el("div", {"class": "bars"});
  LABELS.forEach(function (label) {
    var value = scores[label] || 0;
    var fill = el("div", {"class": "fill"});
    fill.style.width = (value * 100).toFixed(1) + "%";
    wrap.appendChild(el("div", {"class": "bar-row"}, [el("span", {text: label}), el("div", {"class": "track"}, [fill]), el("span", {text: pct(value)})]));
  });
  return wrap;
}

function renderSingle(result) {
  var box = clear(document.getElementById("single-result"));
  box.hidden = false;
  if (!result.ok) { box.appendChild(el("p", {"class": "error", text: result.body.message || "Something went wrong."})); return; }
  var answer = result.body;
  var line = el("div", {}, [el("span", {"class": "label", text: answer.label})]);
  if (answer.review) line.appendChild(el("span", {"class": "pill " + answer.review, text: answer.review === "ok" ? "confident" : "needs review"}));
  box.appendChild(line);
  box.appendChild(scoreBars(answer.scores));
  box.appendChild(el("p", {"class": "note", text: "Model " + answer.model_version + ", " + answer.inference_ms + " ms. The score is the model's confidence, not a calibrated probability."}));
}

function classifyOne(event) {
  event.preventDefault();
  var payload = {title: document.getElementById("title").value, body: document.getElementById("body").value};
  api("/v1/classify", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)}).then(renderSingle);
}

function renderBatch() {
  var summary = clear(document.getElementById("batch-summary"));
  var table = clear(document.getElementById("batch-table"));
  if (!lastBatch) return;
  var counts = {all: lastBatch.length, needs_review: 0, error: 0};
  LABELS.forEach(function (label) { counts[label] = 0; });
  lastBatch.forEach(function (row) {
    if (row.error) counts.error += 1; else counts[row.label] += 1;
    if (row.review === "needs_review") counts.needs_review += 1;
  });
  ["all"].concat(LABELS, ["needs_review", "error"]).forEach(function (name) {
    var chip = el("button", {type: "button", "class": "chip", "aria-pressed": String(filter === name), text: name.replace("_", " ") + " " + counts[name]});
    chip.addEventListener("click", function () { filter = name; renderBatch(); });
    summary.appendChild(chip);
  });
  summary.hidden = false;
  var head = el("tr", {}, [el("th", {text: "Row"}), el("th", {text: "Title"}), el("th", {text: "Label"}), el("th", {"class": "num", text: "Score"}), el("th", {text: "Review"})]);
  var body = el("tbody");
  lastBatch.filter(function (row) {
    if (filter === "all") return true;
    if (filter === "error") return !!row.error;
    if (filter === "needs_review") return row.review === "needs_review";
    return row.label === filter;
  }).forEach(function (row) {
    body.appendChild(el("tr", {}, [el("td", {text: String(row.row)}), el("td", {text: row.title}),
      el("td", {text: row.error ? "error: " + row.error : row.label}), el("td", {"class": "num", text: row.error ? "" : pct(row.score)}),
      el("td", {text: row.review === "needs_review" ? "needs review" : ""})]));
  });
  table.appendChild(el("table", {}, [el("thead", {}, [head]), body]));
  table.hidden = false;
}

function readFile() {
  var file = document.getElementById("file").files[0];
  return file ? file.text() : Promise.resolve(null);
}

function classifyBatch(event) {
  event.preventDefault();
  readFile().then(function (text) {
    if (text === null) return;
    api("/v1/classify-batch", {method: "POST", headers: {"content-type": "text/csv"}, body: text}).then(function (result) {
      var table = clear(document.getElementById("batch-table"));
      if (!result.ok) {
        document.getElementById("batch-summary").hidden = true;
        table.hidden = false;
        table.style.padding = "20px";
        table.appendChild(el("p", {"class": "error", text: result.body.message || "Something went wrong."}));
        document.getElementById("download").disabled = true;
        return;
      }
      table.style.padding = "";
      lastBatch = result.body.results;
      filter = "all";
      document.getElementById("download").disabled = false;
      renderBatch();
    });
  });
}

function downloadBatch() {
  readFile().then(function (text) {
    if (text === null) return;
    fetch("/v1/classify-batch?format=csv", {method: "POST", headers: {"content-type": "text/csv"}, body: text}).then(function (response) {
      return response.blob();
    }).then(function (blob) {
      var link = el("a", {href: URL.createObjectURL(blob), download: "triage.csv"});
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  });
}

function stat(value, caption) { return el("div", {"class": "stat"}, [el("b", {text: value}), el("span", {text: caption})]); }

function titled(title, table) {
  return el("div", {"class": "panel"}, [el("h2", {text: title}), el("div", {"class": "table-wrap"}, [table])]);
}

function classTable(report) {
  var body = el("tbody");
  LABELS.forEach(function (label) {
    var row = report.per_class[label];
    body.appendChild(el("tr", {}, [el("td", {text: label}), el("td", {"class": "num", text: fixed(row.precision)}), el("td", {"class": "num", text: fixed(row.recall)}),
      el("td", {"class": "num", text: fixed(row.f1)}), el("td", {"class": "num", text: String(row.support)})]));
  });
  var head = el("tr", {}, ["Class", "Precision", "Recall", "F1", "Rows"].map(function (name, i) { return el("th", {"class": i ? "num" : "", text: name}); }));
  return titled("Per class", el("table", {}, [el("thead", {}, [head]), body]));
}

function matrix(report) {
  var cells = report.confusion_matrix.matrix;
  var grid = el("div", {"class": "matrix"});
  grid.style.gridTemplateColumns = "repeat(" + (LABELS.length + 1) + ", auto)";
  grid.appendChild(el("div", {"class": "head", text: "true \\ predicted"}));
  LABELS.forEach(function (label) { grid.appendChild(el("div", {"class": "head", text: label})); });
  var top = 0;
  cells.forEach(function (row) { row.forEach(function (value) { top = Math.max(top, value); }); });
  cells.forEach(function (row, i) {
    grid.appendChild(el("div", {"class": "head", text: LABELS[i]}));
    row.forEach(function (value, j) {
      var cell = el("div", {text: String(value)});
      var strength = top ? value / top : 0;
      cell.style.background = "color-mix(in srgb, var(--acc) " + Math.round(strength * (i === j ? 80 : 55)) + "%, transparent)";
      grid.appendChild(cell);
    });
  });
  return el("div", {"class": "panel"}, [el("h2", {text: "Confusion matrix (held-out test)"}), grid]);
}

function comparisonTable(comparison) {
  var body = el("tbody");
  var models = comparison.models;
  Object.keys(models).sort(function (a, b) { return models[b].macro_f1 - models[a].macro_f1; }).forEach(function (name) {
    var item = models[name];
    body.appendChild(el("tr", {}, [el("td", {text: name + (name === comparison.chosen ? " (selected)" : "")}), el("td", {"class": "num", text: fixed(item.macro_f1)}),
      el("td", {"class": "num", text: item.train_seconds + " s"})]));
  });
  var head = el("tr", {}, ["Model", "Dev macro-F1", "Training time"].map(function (name, i) { return el("th", {"class": i ? "num" : "", text: name}); }));
  return titled("Models compared on the development data", el("table", {}, [el("thead", {}, [head]), body]));
}

function loadEvaluation() {
  var root = clear(document.getElementById("eval-body"));
  Promise.all([api("/v1/report"), api("/v1/comparison"), api("/v1/model-info")]).then(function (results) {
    var report = results[0], comparison = results[1], info = results[2];
    if (!report.ok) {
      root.appendChild(el("div", {"class": "panel"}, [el("p", {"class": "error", text: "No evaluation is available: " + (report.body.message || "unknown")}),
        el("p", {"class": "note", text: "Run the final test once for this model release to create the report."})]));
    } else {
      var r = report.body;
      var grid = el("div", {"class": "grid3"}, [stat(fixed(r.macro_f1), "macro-F1 on the held-out test set"), stat(fixed(r.cross_repo_mean_f1), "mean of the five per-project F1 scores"),
        stat(fixed(r.accuracy), "accuracy"), stat(r.latency_ms_single.median.toFixed(1) + " ms", "median time for one issue (" + r.environment.machine + ")")]);
      root.appendChild(grid);
      root.appendChild(classTable(r));
      root.appendChild(matrix(r));
      var repoBody = el("tbody");
      Object.keys(r.per_repo_macro_f1).forEach(function (repo) {
        repoBody.appendChild(el("tr", {}, [el("td", {text: repo}), el("td", {"class": "num", text: fixed(r.per_repo_macro_f1[repo])})]));
      });
      root.appendChild(titled("Per project", el("table", {}, [el("thead", {}, [el("tr", {}, [el("th", {text: "Project"}), el("th", {"class": "num", text: "Macro-F1"})])]), repoBody])));
      var review = r.review;
      root.appendChild(el("div", {"class": "panel"}, [el("h2", {text: "Needs-review threshold"}),
        el("p", {text: review.threshold === null ? "No threshold reached the target accuracy on the development data, so no review state is shown." :
          "Chosen on development data. On the test set the model answers " + pct(review.accepted) + " of issues without review, with " + pct(review.accuracy_on_accepted) + " of those correct."})]));
      var mistakes = el("div", {"class": "panel"}, [el("h2", {text: "Most confident mistakes"})]);
      r.confident_mistakes.forEach(function (item) {
        mistakes.appendChild(el("p", {}, [el("strong", {text: item.repo + ": "}), item.title, el("span", {"class": "note", text: "  true " + item.true + ", predicted " + item.predicted + " (" + item.score + ")"})]));
      });
      root.appendChild(mistakes);
    }
    if (comparison.ok) root.appendChild(comparisonTable(comparison.body));
    var limits = el("div", {"class": "panel"}, [el("h2", {text: "Limits"}),
      el("p", {text: "The test set mixes five open source projects that also appear in training, so these numbers say nothing about projects the model has never seen. The labels come from the projects' own labelling and contain noise. The score is not a calibrated probability."})]);
    if (info.ok) limits.appendChild(el("p", {"class": "note", text: "Release " + info.body.model_version + ", data source " + info.body.dataset.source + " at commit " + info.body.dataset.commit.slice(0, 7) + "."}));
    root.appendChild(limits);
  });
}

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("single").addEventListener("submit", classifyOne);
  document.getElementById("upload").addEventListener("submit", classifyBatch);
  document.getElementById("download").addEventListener("click", downloadBatch);
  document.getElementById("sample").addEventListener("click", function () {
    var example = EXAMPLES[nextExample % EXAMPLES.length];
    nextExample += 1;
    document.getElementById("title").value = example.title;
    document.getElementById("body").value = example.body;
  });
  document.getElementById("theme").addEventListener("click", function () {
    var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });
  window.addEventListener("hashchange", function () { show(location.hash.slice(1)); });
  show(location.hash.slice(1));
  loadStatus();
});
