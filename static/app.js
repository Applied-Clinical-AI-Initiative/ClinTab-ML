/* ClinTAB-ML front end.
   Vanilla JS only: tab nav, fetch calls, table/plot rendering. PapaParse is
   used solely for the instant in-browser preview; the raw CSV still goes to
   Flask, which does all real parsing, stats and ML. */

const State = { columns: [], coltypes: {}, confirmed: false, lastTrain: null };

// ---------- tiny helpers ----------
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, html) => { const e = document.createElement(tag);
  Object.assign(e, props); if (html != null) e.innerHTML = html; return e; };
function toast(msg, kind = "") {
  const t = $("toast"); t.className = "toast " + kind; t.textContent = msg; t.style.display = "block";
  clearTimeout(toast._t); toast._t = setTimeout(() => t.style.display = "none", 5200);
}
function showBusy(msg, sub, withBar) {
  $("busyMsg").textContent = msg || "Working…"; $("busySub").textContent = sub || "";
  $("busyBarWrap").style.display = withBar === false ? "none" : "block";
  setBusyBar(0); $("busy").classList.add("show");
}
function busyMsg(msg, sub) { $("busyMsg").textContent = msg; if (sub != null) $("busySub").textContent = sub; }
function setBusyBar(p) { $("busyBar").style.width = p + "%"; }
function hideBusy() { $("busy").classList.remove("show"); }
async function jget(url) { const r = await fetch(url); return r.json(); }
async function jpost(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  return r.json();
}
function pill(t) { return `<span class="pill ${t}">${t}</span>`; }
function num(v) { return (v === null || v === undefined || v === "") ? "" : v; }

// ---------- hover-definition glossary (Data Summary, Model Training, Model
// Testing, Spline, Clinical Epi only) ----------
const GLOSSARY = {
  "AUROC": "Area under the ROC curve — how well the model tells apart who will vs won't have the outcome. 0.5 = random guessing, 1.0 = perfect.",
  "AUPRC": "Area under the precision-recall curve — like AUROC, but more informative when the outcome is rare.",
  "Sensitivity": "Of patients who truly had the outcome, the % the model correctly caught. Also called recall.",
  "Specificity": "Of patients who truly did not have the outcome, the % correctly cleared.",
  "PPV": "Positive predictive value — when the model predicts “yes,” how often it's actually right. Also called precision.",
  "NPV": "Negative predictive value — when the model predicts “no,” how often it's actually right.",
  "F1": "A single score balancing precision and sensitivity.",
  "F2": "Like F1, but weights sensitivity (catching true cases) higher than precision.",
  "Brier": "How well-calibrated the predicted probabilities are — lower is better.",
  "Accuracy": "The % of all predictions the model got right.",
  "BalancedAccuracy": "Accuracy adjusted so rare classes count as much as common ones.",
  "MacroF1": "F1 averaged evenly across every class, regardless of how common each one is.",
  "R2": "The % of variation in the outcome explained by the model.",
  "RMSE": "Typical prediction error size, in the outcome's own units — penalizes large misses more than small ones.",
  "MAE": "Typical prediction error size, in the outcome's own units, treating all misses equally.",
  "Threshold": "The predicted-probability cutoff above which a prediction counts as “yes” (default 50%).",
  "N": "Number of rows the statistic was computed on.",
  "SD": "Standard deviation — how spread out the values are around the mean.",
  "IQR": "Interquartile range — the span covering the middle 50% of values.",
  "Grid search": "Automatically trying different model settings (hyperparameters) to find the combination that scores best.",
  "SMOTE": "A technique for rare outcomes: creates synthetic extra examples of the minority class so the model doesn't just learn to predict the common outcome.",
  "Confounding variables": "Variables forced into the model and tagged, rather than dropped or left to compete for selection.",
  "Restricted Cubic Spline": "A smooth curve, allowed to bend only at its knots, fit through the data instead of a straight line.",
  "Knots": "The points along the predictor's range where the spline's curve is allowed to change shape.",
  "AIC": "A score for comparing model fit — lower generally means a better-fitting model.",
  "Odds Ratio": "How much more likely the outcome is, in odds terms, for the exposed group vs. the unexposed group.",
  "Risk Ratio": "How much more likely the outcome is, in plain risk/probability terms, for the exposed group.",
  "NNT": "Number needed to treat — how many patients you'd need to treat to prevent one additional bad outcome.",
  "NNH": "Number needed to harm — how many patients you'd need to expose to cause one additional bad outcome.",
  "Kaplan–Meier": "A curve showing the fraction of patients who have not yet had an event (like death) over time.",
  "Cox proportional hazards": "A model estimating how much various factors speed up or slow down time-to-event, as hazard ratios.",
  "Log-rank": "A statistical test comparing two Kaplan–Meier curves to see if they're really different.",
  "concordance": "Like AUROC, but for survival models — how often the model ranks patients' event risk in the right order.",
  "HR": "Hazard ratio — how much a covariate speeds up (>1) or slows down (<1) time to the event.",
  "Hosmer–Lemeshow": "A test checking whether a model's predicted probabilities match the real observed event rates.",
  "HL statistic": "The Hosmer–Lemeshow test statistic — larger values suggest worse calibration.",
  "2×2 table": "The four-cell table of exposed/unexposed crossed with outcome yes/no, used for OR/RR/NNT calculations.",
  "Pearson r": "A number from -1 to 1 measuring the strength and direction of a straight-line relationship between a feature and the outcome. Near 0 means little linear relationship — there could still be a non-linear one.",
};
function term(label, key) {
  return label;
}

// ---------- tab navigation ----------
function refreshTab(tab) {
  if (tab === "summary") loadSummary();
  if (tab === "train") initTrain();
  if (tab === "test") loadModels();
  if (tab === "spline") initSpline();
  if (tab === "epi") initEpi();
  if (tab === "reports") initReports();
}
$("nav").addEventListener("click", (e) => {
  const a = e.target.closest("a"); if (!a) return;
  document.querySelectorAll(".nav a").forEach(x => x.classList.remove("active"));
  a.classList.add("active");
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  $("tab-" + a.dataset.tab).classList.add("active");
  window.scrollTo(0, 0);
  refreshTab(a.dataset.tab);
});
function goToUpload() { document.querySelector('[data-tab="upload"]').click(); }
function goToTrain() {
  document.querySelector('[data-tab="summary"]').classList.add("done");
  document.querySelector('[data-tab="train"]').click();
}

// =====================================================================
// 1. UPLOAD
// =====================================================================
const drop = $("drop"), fileInput = $("fileInput");
drop.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) { toast("Please choose a .csv file.", "err"); return; }
  // instant in-browser preview (local, does not touch the server)
  Papa.parse(file, { header: true, preview: 20, skipEmptyLines: true,
    complete: (res) => renderPreview(res.meta.fields, res.data), error: () => {} });

  // send the raw file to the local Python process (localhost) for real parsing
  const fd = new FormData(); fd.append("file", file);
  const sizeMB = (file.size / 1048576).toFixed(1);
  showBusy(`Uploading ${file.name}`, `${sizeMB} MB`);
  $("uploadInfo").textContent = `Uploading ${file.name} (${sizeMB} MB)…`;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.timeout = 0;
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const p = Math.round(100 * e.loaded / e.total);
    setBusyBar(p);
    if (p >= 100) busyMsg("Processing on server…", "detecting column types (stays on your machine)");
  };
  xhr.onload = () => {
    hideBusy();
    let d;
    try { d = JSON.parse(xhr.responseText); }
    catch (e) {
      const msg = `Server returned an unexpected response (HTTP ${xhr.status}). Check the terminal running app.py.`;
      toast("Upload failed: " + msg, "err"); $("uploadInfo").textContent = msg; return;
    }
    if (xhr.status !== 200 || d.error) {
      const msg = d.error || `HTTP ${xhr.status}`;
      toast("Upload failed: " + msg, "err"); $("uploadInfo").textContent = msg; return;
    }
    State.columns = d.columns; State.confirmed = false;
    $("uploadInfo").innerHTML = `<b>${d.filename}</b> — ${d.n_rows} rows × ${d.n_cols} columns`;
    $("sessionTag").innerHTML = `Dataset: <b style="color:#fff">${d.filename}</b><br>${d.n_rows} rows`;
    renderTypeTable(d.columns); buildStratOptions(d.columns);
    $("typePanel").style.display = $("splitPanel").style.display = "block";
    loadDatasetList();
    toast("Loaded · columns detected.", "ok");
  };
  xhr.onerror = () => {
    hideBusy();
    toast("Cannot reach the server. Is it running?  Start it with:  python app.py", "err");
    $("uploadInfo").innerHTML = '<span class="err-text">Cannot reach the server. Make sure <b>python app.py</b> (or ./run.sh) is running, then reload this page.</span>';
  };
  xhr.ontimeout = () => { hideBusy(); toast("Upload timed out.", "err"); };
  xhr.send(fd);
}

function renderPreview(fields, rows) {
  $("previewPanel").style.display = "block";
  $("previewMeta").textContent = `(first ${rows.length} rows, in-browser)`;
  const t = $("previewTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, fields.map(f => `<th>${f}</th>`).join("")));
  rows.forEach(r => t.appendChild(el("tr", {}, fields.map(f => `<td>${num(r[f])}</td>`).join(""))));
}

const TYPES = ["binary", "categorical", "continuous", "date", "exclude"];
function defaultMissingDecision(pct) {
  if (pct > 50) return "remove";
  if (pct >= 30) return "zero";
  return "impute"; // 0 < pct < 30
}
function missingSelectHtml(c) {
  if (!(c.pct_missing > 0)) return `<span class="muted small">—</span>`;
  const def = defaultMissingDecision(c.pct_missing);
  const opt = (v, label) => `<option value="${v}"${def === v ? " selected" : ""}>${label}</option>`;
  return `<select id="miss_${c.name}" class="small">
      ${opt("impute", "Impute (median / most frequent)")}
      ${opt("zero", "Populate 0")}
      ${opt("include", "Include as-is")}
      ${opt("remove", "Remove column")}
    </select>`;
}
function renderTypeTable(cols) {
  const t = $("typeTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, `<th>Column</th><th>Detected</th><th>Unique</th><th>% missing</th>
    <th>Type override</th><th>Missing-data handling</th>`));
  cols.forEach(c => {
    const tr = el("tr");
    const radios = TYPES.map(tp => `<label class="small" style="font-weight:500;margin-right:8px">
        <input type="radio" name="ty_${c.name}" value="${tp}" ${c.type === tp ? "checked" : ""}> ${tp}</label>`).join("");
    tr.innerHTML = `<td><b>${c.name}</b></td><td>${pill(c.type)}</td><td>${c.n_unique}</td>
      <td ${c.high_missing ? 'style="color:var(--warn);font-weight:600"' : ""}>${c.pct_missing}%</td>
      <td>${radios}</td><td>${missingSelectHtml(c)}</td>`;
    t.appendChild(tr);
  });
}

function buildStratOptions(cols) {
  const sel = $("stratCol"); sel.innerHTML = `<option value="">— none —</option>`;
  cols.filter(c => c.type === "binary" || c.type === "categorical")
    .forEach(c => sel.appendChild(el("option", { value: c.name }, c.name)));
}
$("stratCol").addEventListener("change", () => {
  $("smoteBox").style.display = $("stratCol").value ? "block" : "none";
});
$("splitMethod").addEventListener("change", e => {
  if (e.target.value === "stratified" && !$("stratCol").value)
    toast("Choose a column to stratify by.", "");
});

$("confirmBtn").addEventListener("click", async () => {
  const types = {}, missing = {};
  State.columns.forEach(c => {
    const r = document.querySelector(`input[name="ty_${c.name}"]:checked`);
    types[c.name] = r ? r.value : c.type;
    const m = $(`miss_${c.name}`); if (m) missing[c.name] = m.value;
  });
  const ratios = [+$("rTrain").value, +$("rVal").value, +$("rTest").value];
  $("confirmMsg").textContent = "Splitting & saving…";
  showBusy("Cleaning & splitting dataset…", "saving train / validation / test partitions", false);
  let d;
  try {
    d = await jpost("/api/confirm", {
      types, missing, ratios, method: $("splitMethod").value,
      stratify_col: $("stratCol").value || null, smote: $("smoteToggle").checked
    });
  } catch (e) { hideBusy(); toast("Cannot reach the server — is python app.py running?", "err"); return; }
  hideBusy();
  if (d.error) { toast(d.error, "err"); $("confirmMsg").textContent = d.error; return; }
  State.confirmed = true; State.coltypes = types;
  document.querySelector('[data-tab="upload"]').classList.add("done");
  $("confirmMsg").innerHTML = `<span class="ok-text">Locked.</span> train ${d.n_train} · val ${d.n_val} · test ${d.n_test} (${d.method})`;
  if (d.smote_hint && d.smote_hint.suggest_smote)
    toast(`Minority class ${(d.smote_hint.minority_fraction*100).toFixed(1)}% — SMOTE recommended.`, "");
  toast("Dataset confirmed and saved.", "ok");
  document.querySelector('[data-tab="summary"]').click();
});

// =====================================================================
// 2. SUMMARY
// =====================================================================
async function loadSummary() {
  const d = await jget("/api/summary");
  if (d.error) { $("summaryEmpty").style.display = "block"; $("summaryBody").style.display = "none";
    $("summaryEmpty").textContent = d.error; return; }
  $("summaryEmpty").style.display = "none"; $("summaryBody").style.display = "block";
  const c = d.card;
  const cards = [["Total N", c.total_n], ["Columns", c.n_columns],
    ["Complete cases", `${c.complete_cases} (${c.pct_complete_cases}%)`]];
  if (c.date_range) cards.push(["Date range", `${c.date_range.start} → ${c.date_range.end}`]);
  $("summaryCards").innerHTML = cards.map(([l, v]) =>
    `<div class="card"><div class="lbl">${l}</div><div class="big">${v}</div></div>`).join("");

  const ct = $("contTable"); ct.innerHTML = "";
  ct.appendChild(el("tr", {}, `<th>Variable</th><th>${term("N")}</th><th>% missing</th><th>Mean</th>
    <th>Median</th><th>${term("SD")}</th><th>${term("IQR")}</th><th>Min</th><th>Max</th>`));
  d.continuous.forEach(r => ct.appendChild(el("tr", {}, `<td>${r.variable}</td><td>${r.n}</td>
    <td>${r.pct_missing}</td><td>${num(r.mean)}</td><td>${num(r.median)}</td><td>${num(r.sd)}</td>
    <td>${num(r.iqr)}</td><td>${num(r.min)}</td><td>${num(r.max)}</td>`)));
  if (!d.continuous.length) ct.appendChild(el("tr", {}, `<td colspan="9" class="muted">None</td>`));

  const kt = $("catTable"); kt.innerHTML = "";
  kt.appendChild(el("tr", {}, `<th>Variable</th><th>Type</th><th>N</th><th>% missing</th><th>Category</th><th>Count</th><th>%</th>`));
  d.categorical.forEach(v => {
    v.categories.forEach((cat, i) => kt.appendChild(el("tr", {}, `
      ${i === 0 ? `<td rowspan="${v.categories.length}"><b>${v.variable}</b></td>
        <td rowspan="${v.categories.length}">${pill(v.type)}</td>
        <td rowspan="${v.categories.length}">${v.n}</td>
        <td rowspan="${v.categories.length}">${v.pct_missing}</td>` : ""}
      <td>${cat.category}</td><td>${cat.count}</td><td>${cat.pct}</td>`)));
  });
  if (!d.categorical.length) kt.appendChild(el("tr", {}, `<td colspan="7" class="muted">None</td>`));
}

// =====================================================================
// 3. TRAIN
// =====================================================================
let TRAIN_META = null;
async function initTrain() {
  const d = await jget("/api/train/columns");
  if (d.error) { $("trainEmpty").style.display = "block"; $("trainBody").style.display = "none";
    $("trainEmpty").textContent = d.error; return; }
  TRAIN_META = d;
  $("trainEmpty").style.display = "none"; $("trainBody").style.display = "block";
  $("smoteTrainBox").style.display = d.smote_available ? "flex" : "none";

  const o = $("outcomeSel"); o.innerHTML = "";
  d.outcomes.forEach(c => o.appendChild(el("option", { value: c.name }, `${c.name} (${c.type})`)));
  fillChecklist($("excludeSel"), d.columns.map(c => c.name));
  fillChecklist($("confoundSel"), d.columns.map(c => c.name));
  o.onchange = onOutcomeChange; onOutcomeChange();
}
async function useCustomSplit() {
  const tf = $("customTrainCsv").files[0], vf = $("customValCsv").files[0];
  if (!tf || !vf) { toast("Choose both a train CSV and a validation CSV.", "err"); return; }
  const fd = new FormData();
  fd.append("train_file", tf); fd.append("val_file", vf);
  showBusy("Uploading custom split…", "re-detecting column types");
  const d = await fetch("/api/train/upload-split", { method: "POST", body: fd }).then(r => r.json());
  hideBusy();
  if (d.error) {
    toast(d.error, "err");
    $("customSplitMsg").innerHTML = `<span class="err-text">${d.error}</span>`;
    return;
  }
  $("customSplitMsg").innerHTML = `<span class="ok-text">Using custom split.</span> train ${d.n_train} · val ${d.n_val}`;
  toast("Custom train/validation split applied.", "ok");
  initTrain();
}
function fillMulti(sel, names) { sel.innerHTML = ""; names.forEach(n => sel.appendChild(el("option", { value: n }, n))); }
function fillChecklist(box, names) {
  box.innerHTML = "";
  names.forEach(n => box.appendChild(el("label", {}, `<input type="checkbox" value="${n}"> ${n}`)));
}
function multiValues(box) { return [...box.querySelectorAll("input:checked")].map(c => c.value); }

function taskForOutcome(name) {
  const c = TRAIN_META.outcomes.find(x => x.name === name);
  if (!c) return "binary";
  if (c.type === "continuous") return "continuous";
  if (c.type === "binary") return "binary";
  return "multiclass";
}
function onOutcomeChange() {
  const out = $("outcomeSel").value, task = taskForOutcome(out);
  $("taskHint").innerHTML = `Detected task: <b>${task}</b>`;
  $("modelTaskLbl").textContent = `(${task === "continuous" ? "regressors" : "classifiers"})`;
  // scoring options
  const scores = task === "continuous"
    ? [["r2", "R²"], ["mae", "MAE"], ["rmse", "RMSE"]]
    : [["roc", "ROC AUC (default)"], ["auprc", "AUPRC"], ["f1", "F1"], ["f2", "F2"],
       ["recall", "Recall/Sensitivity"], ["precision", "Precision"], ["accuracy", "Accuracy"]];
  $("scoreSel").innerHTML = scores.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  // models + grids
  const grids = task === "continuous" ? TRAIN_META.default_grids_continuous : TRAIN_META.default_grids_binary;
  const ml = $("modelList"); ml.innerHTML = "";
  const ge = $("gridEditors"); ge.innerHTML = "";
  Object.keys(grids).forEach(m => {
    ml.appendChild(el("label", {}, `<input type="checkbox" class="mdl" value="${m}" checked> ${m}`));
    const id = "grid_" + m;
    ge.appendChild(el("div", {}, `<label class="small" style="margin-top:6px">${m}</label>
      <textarea id="${id}" rows="2">${JSON.stringify(grids[m])}</textarea>`));
  });
  // exclude outcome from feature multiselects (visually keep but warn server handles)
}
function toggleModels(on) { document.querySelectorAll(".mdl").forEach(c => c.checked = on); }

const LARGE_DATASET_ROWS = 50000;
function confirmLargeDataset(nRows) {
  return new Promise(resolve => {
    const modal = $("trainWarnModal");
    $("trainWarnMsg").textContent =
      `Your training set has ${nRows.toLocaleString()} rows. Grid search across multiple models on ` +
      `a dataset this size can take a long time to finish on a local machine. You can proceed and let ` +
      `it run, or cancel and narrow down the models or rows first.`;
    modal.classList.add("show");
    const finish = (v) => {
      modal.classList.remove("show");
      $("trainWarnProceed").removeEventListener("click", onProceed);
      $("trainWarnCancel").removeEventListener("click", onCancel);
      modal.removeEventListener("click", onBackdrop);
      resolve(v);
    };
    const onProceed = () => finish(true);
    const onCancel = () => finish(false);
    const onBackdrop = (e) => { if (e.target === modal) finish(false); };
    $("trainWarnProceed").addEventListener("click", onProceed);
    $("trainWarnCancel").addEventListener("click", onCancel);
    modal.addEventListener("click", onBackdrop);
  });
}

function showConfirm(title, message, okLabel = "Delete") {
  return new Promise(resolve => {
    const modal = $("confirmModal");
    $("confirmTitle").textContent = title;
    $("confirmMsg").textContent = message;
    $("confirmOk").textContent = okLabel;
    modal.classList.add("show");
    const finish = (v) => {
      modal.classList.remove("show");
      $("confirmOk").removeEventListener("click", onOk);
      $("confirmCancel").removeEventListener("click", onCancel);
      modal.removeEventListener("click", onBackdrop);
      resolve(v);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onBackdrop = (e) => { if (e.target === modal) finish(false); };
    $("confirmOk").addEventListener("click", onOk);
    $("confirmCancel").addEventListener("click", onCancel);
    modal.addEventListener("click", onBackdrop);
  });
}

$("trainBtn").addEventListener("click", startTraining);
async function startTraining() {
  const outcome = $("outcomeSel").value;
  const models = [...document.querySelectorAll(".mdl:checked")].map(c => c.value);
  if (!models.length) { toast("Select at least one model.", "err"); return; }
  if (TRAIN_META.n_train && TRAIN_META.n_train > LARGE_DATASET_ROWS) {
    const proceed = await confirmLargeDataset(TRAIN_META.n_train);
    if (!proceed) return;
  }
  const exclude = multiValues($("excludeSel"));
  const confounders = multiValues($("confoundSel"));
  const grids = {};
  models.forEach(m => { const t = $("grid_" + m); if (t) { try { grids[m] = JSON.parse(t.value); } catch (e) {} } });

  $("trainBtn").disabled = true; $("trainLog").style.display = "block"; $("trainLog").textContent = "";
  $("trainResults").style.display = "none"; setBar(0);

  const cfg = await jpost("/api/train", {
    outcome, exclude, confounders, models, grids,
    scoring: $("scoreSel").value, grid_search: $("gridToggle").checked,
    smote: $("smoteTrain").checked, threshold: +$("thrInput").value
  });
  if (cfg.error) { toast(cfg.error, "err"); $("trainBtn").disabled = false; return; }

  const metricsRows = [], impWrap = $("importanceWrap"); impWrap.innerHTML = "";
  let metricKeys = null;
  const es = new EventSource("/api/train/stream");
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "start") log(`▶ Training ${m.n_models} model(s) · ${m.features} features · task=${m.task}`);
    else if (m.event === "progress") { log(`  … ${m.message}`); setBar(m.pct); }
    else if (m.event === "model_error") log(`  ✗ ${m.model}: ${m.error}`);
    else if (m.event === "model_done") {
      log(`  ✓ ${m.model} saved as ${m.saved_as}`); setBar(m.pct);
      if (!metricKeys) metricKeys = Object.keys(m.metrics);
      metricsRows.push({ model: m.model, ...m.metrics });
      if (m.importance_png) {
        impWrap.appendChild(el("div", { className: "plotwrap" },
          `<img class="plot" src="${m.importance_png}" style="max-width:430px">
           <div class="flex"><button class="ghost sm" onclick="dlPng('${m.importance_png}','${m.model}_importance.png')">PNG</button>
           <button class="ghost sm" onclick='dlImpCsv(${JSON.stringify(JSON.stringify(m.importance))},"${m.model}_importance.csv")'>CSV</button></div>`));
      }
    } else if (m.event === "complete") {
      es.close(); $("trainBtn").disabled = false; setBar(100);
      const nOk = m.results.length, nTotal = models.length;
      if (nOk > 0) {
        renderValMetrics(metricKeys, metricsRows);
        $("trainResults").style.display = "block";
        document.querySelector('[data-tab="train"]').classList.add("done");
      }
      toast(
        nOk === nTotal ? "Training complete." : nOk > 0 ? `${nOk} of ${nTotal} models trained.` : "All models failed to train.",
        nOk === nTotal ? "ok" : nOk > 0 ? "" : "err"
      );
      showTrainDone(nOk, nTotal);
    }
  };
  es.onerror = () => { es.close(); $("trainBtn").disabled = false; toast("Training stream error.", "err"); };
}
function showTrainDone(nOk, nTotal) {
  const modal = $("trainDoneModal"), icon = $("trainDoneIcon"), failed = nOk === 0;
  icon.textContent = failed ? "!" : "✓";
  icon.classList.toggle("err", failed);
  $("trainDoneTitle").textContent = failed ? "Training failed" : "Training complete";
  $("trainDoneTest").style.display = failed ? "none" : "";
  $("trainDoneRedo").textContent = failed ? "Close" : "View metrics";
  $("trainDoneMsg").textContent = failed
    ? `None of the ${nTotal} model${nTotal === 1 ? "" : "s"} trained successfully. Check the log above ` +
      `for what went wrong, then adjust your settings and try again.`
    : nOk === nTotal
      ? `All ${nTotal} model${nTotal === 1 ? "" : "s"} trained successfully. Review the validation ` +
        `metrics below, or jump straight to testing.`
      : `${nOk} of ${nTotal} model${nTotal === 1 ? "" : "s"} trained successfully. Check the log above ` +
        `for what went wrong with the rest.`;
  modal.classList.add("show");
  const close = () => modal.classList.remove("show");
  const viewMetrics = () => { close(); if (!failed) $("trainResults").scrollIntoView({ behavior: "smooth", block: "start" }); };
  $("trainDoneRedo").onclick = viewMetrics;
  $("trainDoneTest").onclick = () => { close(); document.querySelector('[data-tab="test"]').click(); };
  modal.onclick = (e) => { if (e.target === modal) close(); };
}
function log(s) { const l = $("trainLog"); l.textContent += s + "\n"; l.scrollTop = l.scrollHeight; }
function setBar(p) { $("trainBar").style.width = p + "%"; $("trainPct").textContent = p ? p + "%" : ""; }
function renderValMetrics(keys, rows) {
  const t = $("valMetricsTable"); t.innerHTML = "";
  if (!keys || !keys.length) { t.innerHTML = `<tr><td class="muted">No metrics available.</td></tr>`; return; }
  t.appendChild(el("tr", {}, `<th>Model</th>` + keys.map(k => `<th>${term(k)}</th>`).join("")));
  rows.forEach(r => t.appendChild(el("tr", {}, `<td><b>${r.model}</b></td>` +
    keys.map(k => `<td>${num(r[k])}</td>`).join(""))));
}

// =====================================================================
// 4. TEST
// =====================================================================
$("testSource").addEventListener("change", e =>
  $("testUploadBox").style.display = e.target.value === "upload" ? "block" : "none");
$("pklUpload").addEventListener("change", e => {
  if (!e.target.files[0]) return;
  const fd = new FormData(); fd.append("file", e.target.files[0]);
  fetch("/api/model/upload", { method: "POST", body: fd }).then(r => r.json()).then(d => {
    if (d.error) toast(d.error, "err"); else { toast("Model uploaded.", "ok"); loadModels(); }
  });
});
function fmtCreated(ts) {
  if (!ts || ts.length < 15) return "";
  const y = ts.slice(0, 4), mo = ts.slice(4, 6), d = ts.slice(6, 8),
        h = ts.slice(9, 11), mi = ts.slice(11, 13);
  const dt = new Date(`${y}-${mo}-${d}T${h}:${mi}`);
  if (isNaN(dt)) return "";
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    ", " + dt.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}
async function loadModels() {
  const d = await jget("/api/models");
  const list = $("modelPickList"), pm = $("predModel"), hm = $("hlModel");
  if (pm) pm.innerHTML = ""; if (hm) hm.innerHTML = "";
  if (!d.models.length) { list.innerHTML = `<span class="muted small">Train Models to Display for Model Testing.</span>`;
    updateDeleteSelectedBtn(); return; }
  list.innerHTML = "";
  d.models.forEach(m => {
    const meta = m.meta || {};
    const taskPill = meta.task ? `<span class="pill ${meta.task}">${meta.task}</span>` : "";
    const auc = meta.val_metrics && meta.val_metrics.AUROC != null ? `AUROC ${meta.val_metrics.AUROC}` : "";
    const when = fmtCreated(meta.created);
    const metaLine = [auc, when].filter(Boolean).join(" · ");
    const tip = `${m.name}${meta.outcome ? ` — outcome: ${meta.outcome}` : ""}${when ? ` — trained ${when}` : ""}`;
    list.appendChild(el("label", { title: tip }, `<input type="checkbox" class="tmdl" value="${m.name}">
      <span class="mdl-info">
        <div class="mdl-top"><b>${meta.model || m.name}</b>${taskPill}</div>
        ${meta.outcome ? `<span class="muted small mdl-sub">Outcome: ${meta.outcome}</span>` : ""}
        ${metaLine ? `<span class="muted small mdl-sub">${metaLine}</span>` : ""}
      </span>`));
    if (pm) pm.appendChild(el("option", { value: m.name }, m.name));
    if (hm && meta.task === "binary") hm.appendChild(el("option", { value: m.name }, m.name));
  });
  updateDeleteSelectedBtn();
}
function updateDeleteSelectedBtn() {
  const n = document.querySelectorAll(".tmdl:checked").length;
  const btn = $("deleteSelectedBtn");
  btn.style.display = n ? "" : "none";
  btn.textContent = n ? `Delete selected (${n})` : "Delete selected";
}
$("modelPickList").addEventListener("change", (e) => { if (e.target.classList.contains("tmdl")) updateDeleteSelectedBtn(); });
async function confirmDeleteSelected() {
  const names = [...document.querySelectorAll(".tmdl:checked")].map(c => c.value);
  if (!names.length) return;
  const ok = await showConfirm(
    "Delete selected models?",
    `This will permanently delete ${names.length} model${names.length === 1 ? "" : "s"}. This cannot be undone.`,
    "Delete"
  );
  if (!ok) return;
  const results = await Promise.all(names.map(n =>
    fetch(`/api/models/${encodeURIComponent(n)}`, { method: "DELETE" }).then(r => r.json())));
  const failed = results.filter(r => r.error).length;
  toast(failed ? `Deleted ${names.length - failed} of ${names.length} models.` : "Model(s) deleted.",
        failed ? "err" : "ok");
  loadModels();
}
$("testBtn").addEventListener("click", async () => {
  const names = [...document.querySelectorAll(".tmdl:checked")].map(c => c.value);
  if (!names.length) { toast("Pick at least one model.", "err"); return; }
  const source = $("testSource").value;
  const fd = new FormData();
  names.forEach(n => fd.append("models", n));
  fd.append("source", source); fd.append("threshold", $("testThr").value);
  if (source === "upload") {
    if (!$("testCsv").files[0]) { toast("Choose a CSV.", "err"); return; }
    fd.append("file", $("testCsv").files[0]);
  }
  $("testBtn").disabled = true;
  const d = await fetch("/api/test", { method: "POST", body: fd }).then(r => r.json());
  $("testBtn").disabled = false;
  if (d.error) { toast(d.error, "err"); return; }
  document.querySelector('[data-tab="test"]').classList.add("done");
  renderTest(d);
});
function renderTest(d) {
  $("testResults").style.display = "block";
  $("testResults").scrollIntoView({ behavior: "smooth", block: "start" });
  const ov = $("testOverlay"); ov.innerHTML = "";
  if (d.roc_png) ov.appendChild(plotBox(d.roc_png, "ROC overlay", "roc_overlay.png"));
  if (d.pr_png) ov.appendChild(plotBox(d.pr_png, "PR overlay", "pr_overlay.png"));

  const valid = d.models.filter(m => m.metrics);
  const keys = valid.length ? Object.keys(valid[0].metrics) : [];
  const t = $("testMetricsTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, `<th>Model</th>` + keys.map(k => `<th>${term(k)}</th>`).join("")));
  d.models.forEach(m => {
    if (m.error) { t.appendChild(el("tr", {}, `<td><b>${m.name}</b></td><td colspan="${keys.length}" class="err-text">${m.error}</td>`)); return; }
    t.appendChild(el("tr", {}, `<td><b>${m.name}</b></td>` + keys.map(k => `<td>${num(m.metrics[k])}</td>`).join("")));
  });

  const pp = $("testPlots"); pp.innerHTML = "";
  d.models.forEach(m => {
    const hasPlots = m.confusion_png || m.calibration_png || m.residuals_png ||
      m.pred_vs_actual_png || (m.correlations && m.correlations.length);
    if (!hasPlots) return;
    const group = el("div", { className: "test-model-group" });
    group.appendChild(el("h4", {}, m.name));
    const row = el("div", { className: "test-model-row" });
    if (m.confusion_png) row.appendChild(plotBox(m.confusion_png, "confusion", "confusion.png"));
    if (m.calibration_png) row.appendChild(plotBox(m.calibration_png, "calibration", "calibration.png"));
    if (m.residuals_png) row.appendChild(plotBox(m.residuals_png, "residuals", "residuals.png"));
    if (m.pred_vs_actual_png) row.appendChild(plotBox(m.pred_vs_actual_png, "pred vs actual", "pva.png"));
    if (m.correlations && m.correlations.length) row.appendChild(correlationBox(m.name, m.correlations));
    group.appendChild(row);
    pp.appendChild(group);
  });
}
function plotBox(src, label, fname) {
  const d = el("div", { className: "plotwrap" });
  d.innerHTML = `<div class="small muted">${label}</div><img class="plot" src="${src}" style="max-width:430px">
    <div class="flex"><button class="ghost sm" onclick="dlPng('${src}','${fname}')">PNG</button></div>`;
  return d;
}
function correlationBox(name, rows) {
  const d = el("div", { className: "plotwrap" });
  const body = rows.map(r => `<tr><td>${r.feature}</td><td>${num(r.pearson_r)}</td></tr>`).join("");
  d.innerHTML = `<details>
      <summary>${name}: view ${term("Pearson r")} table</summary>
      <table style="margin-top:6px;max-width:320px">
        <tr><th>Feature</th><th>${term("Pearson r")}</th></tr>${body}</table>
    </details>`;
  return d;
}

// single prediction
async function buildPredForm() {
  const name = $("predModel").value; if (!name) { toast("Pick a model.", "err"); return; }
  const d = await jget("/api/model/features?name=" + encodeURIComponent(name));
  const w = $("predFormWrap"); w.innerHTML = "";
  if (!d.features) { w.innerHTML = `<span class="muted small">Uploaded model has no stored feature list — use the JSON upload below.</span>`; return; }
  const grid = el("div", { className: "checklist" });
  d.features.forEach(f => grid.appendChild(el("div", {}, `<label>${f}</label><input class="pf" data-f="${f}" type="text">`)));
  w.appendChild(grid);
  w.appendChild(el("div", { style: "margin-top:10px" }, `<button onclick="predictForm('${name}')">Predict outcome</button>`));
}
async function predictForm(name) {
  const row = {};
  document.querySelectorAll(".pf").forEach(i => { if (i.value !== "") row[i.dataset.f] = isNaN(+i.value) ? i.value : +i.value; });
  const d = await jpost("/api/predict-single", { model: name, row });
  showPred(d);
}
async function predictUpload() {
  if (!$("predPkl").files[0] || !$("predJson").files[0]) { toast("Choose model and JSON.", "err"); return; }
  const fd = new FormData(); fd.append("model_file", $("predPkl").files[0]); fd.append("json_file", $("predJson").files[0]);
  const d = await fetch("/api/predict-single", { method: "POST", body: fd }).then(r => r.json());
  showPred(d);
}
function showPred(d) {
  if (d.error) { toast(d.error, "err"); return; }
  let h = `<div class="card"><div class="lbl">Prediction</div><div class="big">${d.prediction}</div>`;
  if (d.probability != null) h += `<div class="small">probability ${(d.probability*100).toFixed(1)}%</div>`;
  h += `<div class="small muted" style="margin-top:6px">${d.explanation}</div></div>`;
  $("predResult").innerHTML = h;
}

// =====================================================================
// 5. SPLINE
// =====================================================================
async function initSpline() {
  const d = await jget("/api/train/columns");
  if (d.error) { $("splineEmpty").style.display = "block"; $("splineBody").style.display = "none"; $("splineEmpty").textContent = d.error; return; }
  $("splineEmpty").style.display = "none"; $("splineBody").style.display = "block";
  fillSel($("splPred"), d.columns.filter(c => c.type === "continuous").map(c => c.name));
  fillSel($("splOut"), d.columns.filter(c => c.type === "binary").map(c => c.name));
}
$("splBtn").addEventListener("click", async () => {
  const d = await jpost("/api/spline", { predictor: $("splPred").value, outcome: $("splOut").value, n_knots: +$("splKnots").value });
  if (d.error) { toast(d.error, "err"); return; }
  document.querySelector('[data-tab="spline"]').classList.add("done");
  $("splineResult").innerHTML = `<img class="plot" src="${d.png}" style="max-width:560px">
    <div class="flex" style="margin-top:6px">
      <button class="ghost sm" onclick="dlPng('${d.png}','spline.png')">PNG</button>
      <span class="small muted">N=${d.n} · ${term("knots", "Knots")} at ${d.knots.join(", ")} · ${term("AIC")} ${d.aic}</span></div>`;
});

// =====================================================================
// 6. EPI
// =====================================================================
async function initEpi() {
  const d = await jget("/api/train/columns");
  if (d.error) { $("epiEmpty").style.display = "block"; $("epiBody").style.display = "none"; $("epiEmpty").textContent = d.error; return; }
  $("epiEmpty").style.display = "none"; $("epiBody").style.display = "block";
  const allNum = d.columns.map(c => c.name);
  fillSel($("kmTime"), allNum); fillSel($("kmEvent"), allNum);
  fillSel($("kmGroup"), allNum, true); fillMulti($("coxCov"), allNum);
  loadModels();
}
function fillSel(sel, names, withNone) {
  sel.innerHTML = withNone ? `<option value="">— none —</option>` : "";
  names.forEach(n => sel.appendChild(el("option", { value: n }, n)));
}
async function runEpi2x2() {
  const body = { a: +$("ea").value, b: +$("eb").value, c: +$("ec").value, d: +$("ed").value };
  const r = await jpost("/api/epi/or", body);
  const n = await jpost("/api/epi/nnt", body);
  if (r.error) { toast(r.error, "err"); return; }
  $("epi2x2Result").innerHTML = `
    <div class="cards">
      <div class="card"><div class="lbl">${term("Odds Ratio")}</div><div class="big">${r.odds_ratio}</div><div class="small">95% CI ${r.or_ci[0]}–${r.or_ci[1]}</div></div>
      <div class="card"><div class="lbl">${term("Risk Ratio")}</div><div class="big">${r.risk_ratio}</div><div class="small">95% CI ${r.rr_ci[0]}–${r.rr_ci[1]}</div></div>
      <div class="card"><div class="lbl">${term(n.measure || "NNT/NNH", n.measure)}</div><div class="big">${n.value ?? "—"}</div><div class="small">ARD ${r.abs_risk_diff}</div></div>
      <div class="card"><div class="lbl">χ² p-value</div><div class="big">${r.p_value}</div><div class="small">χ²=${r.chi2}</div></div>
    </div>
    <div class="small muted" style="margin-top:8px">${n.interpretation || ""} ${r.corrected ? "(Haldane–Anscombe correction applied)" : ""}</div>`;
}
async function runKM() {
  const d = await jpost("/api/epi/km", { time: $("kmTime").value, event: $("kmEvent").value, group: $("kmGroup").value || null });
  if (d.error) { toast(d.error, "err"); return; }
  let h = `<img class="plot" src="${d.png}" style="max-width:560px">
    <div class="flex" style="margin-top:6px"><button class="ghost sm" onclick="dlPng('${d.png}','kaplan_meier.png')">PNG</button></div>
    <table style="margin-top:8px;width:auto"><tr><th>Group</th><th>N</th><th>Median survival</th></tr>`;
  d.curves.forEach(c => h += `<tr><td>${c.name}</td><td>${c.n}</td><td>${c.median_survival ?? "not reached"}</td></tr>`);
  h += `</table>`;
  if (d.logrank) h += `<div class="small" style="margin-top:6px">${term("Log-rank")} χ²=${d.logrank.test_statistic}, p=${d.logrank.p_value}</div>`;
  $("kmResult").innerHTML = h;
}
async function runCox() {
  const cov = [...$("coxCov").selectedOptions].map(o => o.value);
  if (!cov.length) { toast("Pick covariates.", "err"); return; }
  const d = await jpost("/api/epi/cox", { time: $("kmTime").value, event: $("kmEvent").value, covariates: cov });
  if (d.error) { toast(d.error, "err"); return; }
  let h = `<div class="small muted">N=${d.n}, events=${d.n_events}, ${term("concordance")}=${d.concordance}</div>
    <table style="margin-top:6px"><tr><th>Covariate</th><th>${term("HR")}</th><th>95% CI</th><th>coef</th><th>p</th></tr>`;
  d.rows.forEach(r => h += `<tr><td>${r.covariate}</td><td>${r.hazard_ratio}</td><td>${r.hr_lower}–${r.hr_upper}</td><td>${r.coef}</td><td>${r.p_value}</td></tr>`);
  $("coxResult").innerHTML = h + `</table>`;
}
async function runHL() {
  if (!$("hlModel").value) { toast("Pick a binary model.", "err"); return; }
  const d = await jpost("/api/epi/hl", { model: $("hlModel").value });
  if (d.error) { toast(d.error, "err"); return; }
  let h = `<div class="cards"><div class="card"><div class="lbl">${term("HL statistic")}</div><div class="big">${d.hl_statistic}</div>
      <div class="small">df ${d.dof}</div></div>
      <div class="card"><div class="lbl">p-value</div><div class="big">${d.p_value}</div>
      <div class="small">${d.well_calibrated ? "well calibrated" : "poor fit"}</div></div></div>
    <img class="plot" src="${d.png}" style="max-width:480px;margin-top:10px"><div class="flex" style="margin-top:6px">
    <button class="ghost sm" onclick="dlPng('${d.png}','hosmer_lemeshow.png')">PNG</button></div>
    <table style="margin-top:8px"><tr><th>Decile</th><th>N</th><th>Mean pred</th><th>Observed rate</th><th>Obs</th><th>Exp</th></tr>`;
  d.deciles.forEach((x, i) => h += `<tr><td>${i+1}</td><td>${x.n}</td><td>${x.mean_pred}</td><td>${x.observed_rate}</td><td>${x.observed}</td><td>${x.expected}</td></tr>`);
  $("hlResult").innerHTML = h + `</table>`;
}

// =====================================================================
// 7. REPORTS
// =====================================================================
async function initReports() {
  const d = await jget("/api/session");
  const confirmed = d.active && d.meta && d.meta.stage === "confirmed";
  $("reportsEmpty").style.display = confirmed ? "none" : "block";
  $("reportsBody").style.display = confirmed ? "block" : "none";
  if (!confirmed) return;
  loadMethodsText(); loadTripodReport();
}
async function loadMethodsText() {
  const d = await jget("/api/report/methods");
  if (d.error) { toast(d.error, "err"); return; }
  $("methodsTextArea").textContent = d.text || "";
}
function copyMethodsText() {
  const text = $("methodsTextArea").textContent;
  if (!text) { toast("Nothing to copy yet.", "err"); return; }
  navigator.clipboard.writeText(text).then(() => toast("Copied.", "ok"));
}
function toggleEditMethods() {
  const p = $("methodsTextArea");
  const editing = p.getAttribute("contenteditable") === "true";
  p.setAttribute("contenteditable", editing ? "false" : "true");
  $("editMethodsBtn").textContent = editing ? "Edit" : "Done editing";
  if (!editing) { p.focus(); toast("Editing locally — not saved to the server.", ""); }
}
function expandMethods() {
  $("methodsModalSlot").appendChild($("methodsPaper"));
  $("methodsModal").classList.add("show");
}
function closeMethodsModal() {
  $("methodsPaperSlot").appendChild($("methodsPaper"));
  $("methodsModal").classList.remove("show");
}
async function loadTripodReport() {
  const d = await jget("/api/report/tripod");
  if (d.error) { toast(d.error, "err"); return; }
  $("tripodDisclaimer").textContent = d.disclaimer;
  const s = d.summary;
  $("tripodSummary").innerHTML = [
    ["Found", s.found], ["Not found", s.not_found],
    ["Not applicable", s.not_applicable], ["Manual review", s.manual_review]
  ].map(([l, v]) => `<div class="card"><div class="lbl">${l}</div><div class="big">${v}</div></div>`).join("");
  const t = $("tripodTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, `<th>Section</th><th>Item</th><th>Status</th>`));
  d.items.forEach(i => t.appendChild(el("tr", {}, `<td class="small">${i.section}</td>
    <td class="small">${i.description}</td>
    <td><span class="pill ${i.status}">${i.status.replace(/_/g, " ")}</span></td>`)));
}

// =====================================================================
// downloads
// =====================================================================
function dlPng(dataUrl, fname) { const a = el("a", { href: dataUrl, download: fname }); a.click(); }
function dlBlob(text, fname, type = "text/csv") {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: fname }); a.click();
}
function tableToCsv(table) {
  return [...table.rows].map(r => [...r.cells].map(c =>
    `"${c.textContent.replace(/"/g, '""').trim()}"`).join(",")).join("\n");
}
function dlTable(id, fname) { dlBlob(tableToCsv($(id)), fname); }
function dlImpCsv(jsonStr, fname) {
  const rows = JSON.parse(jsonStr);
  const keys = Object.keys(rows[0] || { feature: "", importance: "" });
  const csv = keys.join(",") + "\n" + rows.map(r => keys.map(k => r[k]).join(",")).join("\n");
  dlBlob(csv, fname);
}

// =====================================================================
// dataset history (sidebar)
// =====================================================================
function applySessionMeta(m) {
  if (!m) { $("sessionTag").textContent = "No dataset loaded"; State.confirmed = false; return; }
  $("sessionTag").innerHTML = `Dataset: <b style="color:#fff">${m.filename || "loaded"}</b><br>${m.n_rows || ""} rows`;
  State.confirmed = m.stage === "confirmed";
  State.coltypes = m.coltypes || {};
  document.querySelector('[data-tab="upload"]').classList.toggle("done", State.confirmed);
}

async function loadDatasetList() {
  const d = await jget("/api/sessions");
  const list = $("datasetList");
  if (!d.sessions || !d.sessions.length) { list.innerHTML = `<div class="ds-empty">No datasets yet.</div>`; return; }
  list.innerHTML = "";
  d.sessions.forEach(s => {
    const active = s.session_id === d.active;
    const label = s.filename || s.session_id;
    const item = el("div", { className: "ds-item" + (active ? " active" : "") });
    item.innerHTML = `<span class="ds-label" title="${label}">${label}${s.n_rows ? ` · ${s.n_rows} rows` : ""}</span>
      <span class="ds-x" title="Delete dataset">&times;</span>`;
    item.querySelector(".ds-label").addEventListener("click", () => switchDataset(s.session_id));
    item.querySelector(".ds-x").addEventListener("click", (e) => { e.stopPropagation(); deleteDataset(s.session_id, label); });
    list.appendChild(item);
  });
}

async function switchDataset(sid) {
  const d = await jpost(`/api/sessions/${sid}/activate`, {});
  if (d.error) { toast(d.error, "err"); return; }
  applySessionMeta(d.meta);
  loadDatasetList();
  const activeTab = document.querySelector(".nav a.active");
  if (activeTab) refreshTab(activeTab.dataset.tab);
  toast("Switched dataset.", "ok");
}

async function deleteDataset(sid, label) {
  const ok = await showConfirm("Delete this dataset?",
    `"${label}" and everything derived from it (splits, models trained on it) will be permanently deleted. This cannot be undone.`,
    "Delete");
  if (!ok) return;
  const r = await fetch(`/api/sessions/${sid}`, { method: "DELETE" }).then(r => r.json());
  if (r.error) { toast(r.error, "err"); return; }
  const sess = await jget("/api/session");
  if (!sess.active) applySessionMeta(null);
  loadDatasetList();
  toast("Dataset deleted.", "ok");
}

$("clearAllBtn").addEventListener("click", async () => {
  const ok = await showConfirm("Clear all data?",
    "This permanently deletes every uploaded dataset on this machine. This cannot be undone.", "Clear all data");
  if (!ok) return;
  const r = await fetch("/api/sessions", { method: "DELETE" }).then(r => r.json());
  if (r.error) { toast(r.error, "err"); return; }
  applySessionMeta(null);
  document.querySelector('[data-tab="upload"]').classList.remove("done");
  loadDatasetList();
  toast("All data cleared.", "ok");
});

// restore session tag on load
jget("/api/session").then(d => applySessionMeta(d.active ? d.meta : null));
loadDatasetList();
