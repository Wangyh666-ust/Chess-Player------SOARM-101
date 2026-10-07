/* SO-ARM101 / Chess Vision — static demo logic. No network, no upload, no live inference. */
(function () {
  "use strict";

  var D = window.DEMO_DATA;

  if (!D || typeof D !== "object" || !D.datasets) {
    document.addEventListener("DOMContentLoaded", function () {
      document.getElementById("fatal").hidden = false;
      document.querySelector("main").setAttribute("hidden", "hidden");
    });
    return;
  }

  var SYMBOLS = {
    K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
    k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F"
  };
  var FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

  /* ---- DOM helpers (textContent-first, no unescaped innerHTML) ---- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function pct2(x) { return pct(x, 2); }
  function pp2(x) { return pp(x, 2); }
  function ms2(x) { return ms(x, 2); }
  function pct(x, digits) {
    if (typeof x !== "number" || !isFinite(x)) return "—";
    return (100 * x).toFixed(digits === undefined ? 1 : digits) + "%";
  }
  function pp(x, digits) {
    if (typeof x !== "number" || !isFinite(x)) return "—";
    var v = 100 * x;
    return (v >= 0 ? "+" : "") + v.toFixed(digits === undefined ? 1 : digits) + " pp";
  }
  function ms(x, digits) {
    if (typeof x !== "number" || !isFinite(x)) return "—";
    return x.toFixed(digits === undefined ? 1 : digits) + " ms";
  }
  function frac(num, den) {
    if (typeof num !== "number" || typeof den !== "number" || !den) return "—";
    return num + " / " + den;
  }
  function datasetLabel(key, ds) {
    return (ds && ds.label) ? ds.label : key;
  }

  /* ---- metric card ---- */
  function metricCard(label, valueText, subNodes) {
    var card = el("div", "metric");
    card.appendChild(el("div", "metric-label", label));
    card.appendChild(el("div", "metric-value", valueText));
    var sub = el("div", "metric-sub");
    (subNodes || []).forEach(function (n) {
      if (typeof n === "string") sub.appendChild(document.createTextNode(n));
      else sub.appendChild(n);
    });
    card.appendChild(sub);
    return card;
  }
  function span(cls, text) { return el("span", cls, text); }

  /* ---- state ---- */
  var state = {
    datasetKey: "independent_test",
    exampleId: null,
    failuresOnly: false,
    showBoxes: true,
    showCorners: true
  };

  var dsSelect = document.getElementById("dataset-select");
  var exSelect = document.getElementById("example-select");
  var failToggle = document.getElementById("failures-only");
  var boxesToggle = document.getElementById("boxes-toggle");
  var cornersToggle = document.getElementById("corners-toggle");
  var prevBtn = document.getElementById("prev-example");
  var nextBtn = document.getElementById("next-example");
  var counter = document.getElementById("preview-counter");
  var metricsBox = document.getElementById("metrics");
  var svgOrig = document.getElementById("svg-original");
  var svgFt = document.getElementById("svg-finetuned");
  var statOrig = document.getElementById("stat-original");
  var statFt = document.getElementById("stat-finetuned");
  var boardTruth = document.getElementById("board-truth");
  var boardOrig = document.getElementById("board-orig");
  var boardFt = document.getElementById("board-ft");
  var errorList = document.getElementById("error-list");
  var fenBox = document.getElementById("fen-box");
  var tableBody = document.querySelector("#comparison-table tbody");

  function ds() { return D.datasets[state.datasetKey]; }

  /* ---- dataset select ---- */
  function buildDatasetOptions() {
    clear(dsSelect);
    Object.keys(D.datasets).forEach(function (key) {
      var o = document.createElement("option");
      o.value = key;
      o.textContent = datasetLabel(key, D.datasets[key]) +
        "  (" + D.datasets[key].count + " images)";
      dsSelect.appendChild(o);
    });
    dsSelect.value = state.datasetKey;
  }

  /* ---- preview example set (union: first 6, 12 evenly spaced, all failures) ---- */
  function previewExamples() {
    return (ds().examples || []).slice();
  }

  function filteredExamples() {
    var list = previewExamples();
    if (state.failuresOnly) list = list.filter(function (e) { return e.finetuned_failed; });
    return list;
  }

  function exampleStatus(e) {
    return (e.finetuned_failed ? "FT fail" : "FT ok") +
      " \u00B7 elev " + (typeof e.elevation === "number" ? e.elevation.toFixed(2) : "—");
  }

  /* ---- example select ---- */
  function buildExampleOptions() {
    var list = filteredExamples();
    clear(exSelect);
    if (!list.length) {
      exSelect.disabled = true;
      return;
    }
    exSelect.disabled = false;
    list.forEach(function (e) {
      var o = document.createElement("option");
      o.value = e.id;
      o.textContent = e.id + "  [" + exampleStatus(e) + "]";
      exSelect.appendChild(o);
    });
  }

  function currentExample() {
    var d = ds();
    var all = d.examples || [];
    for (var i = 0; i < all.length; i++) {
      if (all[i].id === state.exampleId) return all[i];
    }
    return null;
  }

  function chooseDefaultExample() {
    /* for old9 / original dataset: first roster (first example) */
    var d = ds();
    var all = d.examples || [];
    var list = filteredExamples();

    if (state.failuresOnly) {
      if (list.length) { state.exampleId = list[0].id; return; }
      state.exampleId = null; return;
    }
    if (state.datasetKey === "original9") {
      state.exampleId = all.length ? all[0].id : null;
      return;
    }
    /* default: first non-failed heldout example */
    for (var i = 0; i < all.length; i++) {
      if (!all[i].finetuned_failed) { state.exampleId = all[i].id; return; }
    }
    state.exampleId = all.length ? all[0].id : null;
  }

  /* ---- example navigation ---- */
  function navTo(delta) {
    var list = filteredExamples();
    if (!list.length) return;
    var idx = -1;
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === state.exampleId) { idx = i; break; }
    }
    if (idx < 0) idx = 0;
    var next = idx + delta;
    if (next < 0) next = 0;
    if (next >= list.length) next = list.length - 1;
    state.exampleId = list[next].id;
    exSelect.value = state.exampleId;
    updateNavButtons();
    renderExample();
  }

  function updateNavButtons() {
    var list = filteredExamples();
    var idx = -1;
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === state.exampleId) { idx = i; break; }
    }
    prevBtn.disabled = list.length === 0 || idx <= 0;
    nextBtn.disabled = list.length === 0 || idx < 0 || idx >= list.length - 1;
  }

  /* ---- SVG rendering (image + overlays) ---- */
  function svgEl(tag, attrs) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function renderPanel(holder, example, modelKey, showBoxes, showCorners) {
    clear(holder);
    var model = example[modelKey];
    if (!model) {
      holder.appendChild(el("p", "note", "No recorded data for this model."));
      return;
    }
    var w = example.width || 640;
    var h = example.height || 480;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + w + " " + h,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": modelKey + " recorded detections for " + example.id
    });

    svg.appendChild(svgEl("image", {
      href: example.image,
      x: 0, y: 0, width: w, height: h
    }));

    /* detection boxes */
    if (showBoxes && Array.isArray(model.detections)) {
      model.detections.forEach(function (det) {
        var box = det.box || [];
        if (box.length !== 4) return;
        var x = box[0], y = box[1], x2 = box[2], y2 = box[3];
        var rect = svgEl("rect", {
          x: x, y: y, width: Math.max(1, x2 - x), height: Math.max(1, y2 - y),
          "class": "box-rect", rx: 2
        });
        var title = svgEl("title");
        title.textContent = (det.class_name || det.symbol || "?") +
          " \u00B7 conf " + Number(det.confidence).toFixed(2);
        rect.appendChild(title);
        svg.appendChild(rect);

        var lbl = svgEl("text", {
          x: x, y: Math.max(12, y - 3), "class": "box-label",
          style: "font-size:" + Math.round(w * 0.018) +
            "px;paint-order:stroke;stroke:#fffdf8;stroke-width:2px"
        });
        lbl.textContent = (det.symbol || det.class_name || "?") +
          " " + Number(det.confidence).toFixed(2);
        svg.appendChild(lbl);
      });
    }

    /* known corners */
    if (showCorners && Array.isArray(example.corners) && example.corners.length === 4) {
      var pts = example.corners.map(function (p) { return p[0] + "," + p[1]; }).join(" ");
      svg.appendChild(svgEl("polyline", {
        points: pts + " " + example.corners[0][0] + "," + example.corners[0][1],
        "class": "corner-polyline", "fill": "none"
      }));
      var tags = ["TL", "TR", "BR", "BL"];
      example.corners.forEach(function (p, i) {
        var t = svgEl("text", {
          x: p[0] + 4, y: p[1] + 14, "class": "corner-label",
          style: "font-size:" + Math.max(12, Math.round(w * 0.018)) + "px"
        });
        t.textContent = tags[i];
        svg.appendChild(t);
      });
      var note = svgEl("text", { x: 8, y: 20, "class": "corner-tag" });
      note.textContent = "known corners";
      svg.appendChild(note);
    }

    holder.appendChild(svg);
  }

  /* ---- 8x8 boards ---- */
  function buildRankFileLabels(board) {
    var files = el("div", "file-labels");
    FILES.forEach(function (f) { files.appendChild(el("span", null, f)); });
    return files;
  }

  function renderBoard(container, code64, gt64, highlightErrors) {
    clear(container);
    if (!Array.isArray(code64) || code64.length !== 64) {
      container.appendChild(el("p", "note", "No board reconstruction available."));
      return;
    }
    var gt = Array.isArray(gt64) && gt64.length === 64 ? gt64 : null;
    for (var i = 0; i < 64; i++) {
      var rank = Math.floor(i / 8);   /* 0 = rank 8 */
      var file = i % 8;
      var code = code64[i];
      var isDark = (rank + file) % 2 === 1;
      var sq = el("div", "sq " + (isDark ? "dark" : "light"));
      var isErr = false;
      if (highlightErrors && gt && code !== gt[i]) {
        isErr = true;
        sq.className += " err";
      }
      if (code !== "_" && code !== " ") {
        sq.textContent = SYMBOLS[code] || code;
        if (code === code.toUpperCase()) sq.className += " white-piece";
        else sq.className += " black-piece";
      }
      if (file === 0) {
        var rankIndex = el("span", "rank-index", String(8 - rank));
        sq.appendChild(rankIndex);
      }
      var squareName = FILES[file] + (8 - rank);
      var clsName = code === "_" ? "empty" : code;
      var tip = squareName + " \u00B7 " + (isErr ? "GT " + gt[i] + " vs " + code : clsName);
      sq.title = tip;
      sq.setAttribute("aria-label", tip);
      container.appendChild(sq);
    }
    var host = container.parentNode;
    Array.prototype.slice.call(host.querySelectorAll(".file-labels"))
      .forEach(function (old) { host.removeChild(old); });
    host.appendChild(buildRankFileLabels(container));
  }

  /* ---- errors / stats ---- */
  function renderErrorList(example) {
    clear(errorList);
    var gt = example.truth;
    var ft = example.finetuned ? example.finetuned.board : null;
    var bad = [];
    if (Array.isArray(gt) && Array.isArray(ft) && gt.length === 64 && ft.length === 64) {
      for (var i = 0; i < 64; i++) {
        if (gt[i] !== ft[i]) {
          var rank = Math.floor(i / 8);
          var file = i % 8;
          bad.push({ sq: FILES[file] + (8 - rank), gt: gt[i], ft: ft[i] });
        }
      }
    }
    if (!bad.length) {
      var ok = el("li", "all-ok", "All 64 squares match the ground truth.");
      errorList.appendChild(ok);
      return;
    }
    bad.forEach(function (b) {
      var g = b.gt === "_" ? "empty" : b.gt;
      var p = b.ft === "_" ? "empty" : b.ft;
      errorList.appendChild(el("li", null, b.sq + "  " + g + " \u2192 " + p));
    });
  }

  function statLine(model, truth64) {
    /* occupied correct / occupied + board status */
    var occupied = 0, correct = 0;
    var gt = truth64, pred = model && model.board;
    if (Array.isArray(gt) && Array.isArray(pred) && gt.length === 64 && pred.length === 64) {
      for (var i = 0; i < 64; i++) {
        if (gt[i] !== "_") {
          occupied++;
          if (gt[i] === pred[i]) correct++;
        }
      }
      var exact = true;
      for (var j = 0; j < 64; j++) { if (gt[j] !== pred[j]) { exact = false; break; } }
      return "occupied correct " + correct + " / " + occupied + " \u00B7 " +
        (exact ? "complete board" : "board not exact") +
        (model && typeof model.total_ms === "number" ? " \u00B7 " + ms(model.total_ms) : "");
    }
    return "—";
  }

  /* ---- example rendering ---- */
  function renderExample() {
    var list = filteredExamples();
    if (!list.length) {
      counter.textContent = "No examples match the current filter.";
      clear(svgOrig); clear(svgFt);
      clear(boardTruth); clear(boardOrig); clear(boardFt);
      clear(errorList);
      statOrig.textContent = "—"; statFt.textContent = "—";
      fenBox.textContent = "";
      exSelect.disabled = true;
      prevBtn.disabled = true; nextBtn.disabled = true;
      if (state.failuresOnly) {
        svgOrig.appendChild(el("p", "note", "No fine-tuned failures are available in this preview set for the selected dataset."));
      }
      return;
    }
    if (exSelect) exSelect.disabled = false;

    var example = currentExample();
    if (!example) {
      chooseDefaultExample();
      example = currentExample();
    }
    if (!example) {
      counter.textContent = "No example selected.";
      return;
    }
    exSelect.value = example.id;

    var d = ds();
    var idx = list.map(function (e) { return e.id; }).indexOf(example.id);
    counter.textContent = "Preview " + (idx + 1) + " of " + list.length +
      " previewed examples \u00B7 full benchmark: " + d.count + " images \u00B7 selected " + example.id;

    renderPanel(svgOrig, example, "original", state.showBoxes, state.showCorners);
    renderPanel(svgFt, example, "finetuned", state.showBoxes, state.showCorners);

    statOrig.textContent = statLine(example.original, example.truth);
    statFt.textContent = statLine(example.finetuned, example.truth);

    renderBoard(boardTruth, example.truth, example.truth, false);
    renderBoard(boardOrig, example.original ? example.original.board : null, example.truth, true);
    renderBoard(boardFt, example.finetuned ? example.finetuned.board : null, example.truth, true);

    renderErrorList(example);

    fenBox.textContent = example.fen || "(no FEN in data)";
    updateNavButtons();
  }

  /* ---- metrics ---- */
  function metricNodesFor() {
    var d = ds();
    var o = d.original.aggregate;
    var f = d.finetuned.aggregate;
    var ot = d.original.timing;
    var ft = d.finetuned.timing;
    var count = typeof d.count === "number" ? d.count : 0;
    var cards = [];

    cards.push(metricCard(
      "Joint accuracy (colour + type)",
      pct(f.joint_accuracy, 2),
      [
        "Original ",
        span("orig", pct(o.joint_accuracy, 2)),
        " \u00B7 ",
        span("delta-up", pp(f.joint_accuracy - o.joint_accuracy, 2))
      ]
    ));

    cards.push(metricCard(
      "Exact boards (all 64 squares)",
      frac(f.exact_boards, d.count) + " (" + pct(f.exact_boards / d.count, 2) + ")",
      [
        "Original ",
        span("orig", frac(o.exact_boards, d.count) + " (" + pct(o.exact_boards / d.count, 2) + ")")
      ]
    ));

    cards.push(metricCard(
      "Occupancy F1",
      pct(f.occupancy_f1, 2),
      [
        "Original ",
        span("orig", pct2(o.occupancy_f1))
      ]
    ));

    cards.push(metricCard(
      "Median latency (p95)",
      ms(ft.median_ms, 2),
      [
        "p95 ",
        span("orig", ms(ft.p95_ms, 2)),
        " \u00B7 Original ",
        span("orig", ms(ot.median_ms, 2)),
        " \u00B7 n = " + ft.n
      ]
    ));

    return cards;
  }

  function renderMetrics() {
    clear(metricsBox);
    metricNodesFor().forEach(function (c) { metricsBox.appendChild(c); });
  }

  /* ---- comparison table (full set) ---- */
  function renderTable() {
    clear(tableBody);
    var d = ds();
    var o = d.original.aggregate;
    var f = d.finetuned.aggregate;
    var ot = d.original.timing;
    var ft = d.finetuned.timing;

    var caption = document.querySelector("#comparison-table caption");
    if (caption) {
      caption.textContent = "Full evaluation set: " + d.label +
        " (" + d.count + " images)";
    }

    function row(name, ov, fv) {
      var tr = el("tr");
      var th = el("th", null, name);
      th.setAttribute("scope", "row");
      tr.appendChild(th);
      tr.appendChild(el("td", "col-orig", ov));
      tr.appendChild(el("td", "col-ft", fv));
      tableBody.appendChild(tr);
    }

    row("Occupancy F1", pct(o.occupancy_f1, 2), pct(f.occupancy_f1, 2));
    row("Colour accuracy", pct(o.color_accuracy, 2), pct(f.color_accuracy, 2));
    row("Type accuracy", pct(o.type_accuracy, 2), pct(f.type_accuracy, 2));
    row("Joint accuracy (colour + type)",
      pct(o.joint_accuracy, 2), pct(f.joint_accuracy, 2));
    row("Exact boards",
      frac(o.exact_boards, d.count) + " (" + pct(o.exact_boards / d.count, 2) + ")",
      frac(f.exact_boards, d.count) + " (" + pct(f.exact_boards / d.count, 2) + ")");
    row("Median latency",
      ms(ot.median_ms, 2), ms(ft.median_ms, 2));
  }

  /* ---- wiring ---- */
  function resetDataset(key) {
    state.datasetKey = key;
    state.failuresOnly = false;
    failToggle.checked = false;
    chooseDefaultExample();
    buildExampleOptions();
    exSelect.value = state.exampleId || "";
    renderMetrics();
    renderTable();
    renderExample();
  }

  function onDatasetChange() {
    var key = dsSelect.value;
    if (!D.datasets[key]) return;
    resetDataset(key);
  }

  function onExampleChange() {
    state.exampleId = exSelect.value;
    renderExample();
  }

  function onFailuresChange() {
    state.failuresOnly = failToggle.checked;
    chooseDefaultExample();
    buildExampleOptions();
    if (state.exampleId) exSelect.value = state.exampleId;
    renderExample();
  }

  function init() {
    buildDatasetOptions();
    dsSelect.value = state.datasetKey;
    chooseDefaultExample();
    buildExampleOptions();
    exSelect.value = state.exampleId || "";
    renderMetrics();
    renderTable();
    renderExample();

    dsSelect.addEventListener("change", onDatasetChange);
    exSelect.addEventListener("change", onExampleChange);
    failToggle.addEventListener("change", onFailuresChange);
    prevBtn.addEventListener("click", function () { navTo(-1); });
    nextBtn.addEventListener("click", function () { navTo(1); });
    boxesToggle.addEventListener("change", function () {
      state.showBoxes = boxesToggle.checked;
      renderExample();
    });
    cornersToggle.addEventListener("change", function () {
      state.showCorners = cornersToggle.checked;
      renderExample();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
