/* clintab-ml front end.
   vanilla js only: tab nav, fetch calls, table/plot rendering. papaparse is
   used solely for the instant in-browser preview; the raw csv still goes to
   flask, which does all real parsing, stats and ml. */

const State = { columns: [], coltypes: {}, confirmed: false, lastTrain: null, dataVersion: 0 };
// results drawn for one dataset are cleared when another becomes active
function dataChanged() { State.dataVersion++; }
function clearStale(tag, ids) {
  if (clearStale.seen[tag] === State.dataVersion) return;
  clearStale.seen[tag] = State.dataVersion;
  ids.forEach(id => { const e = $(id); if (e) e.innerHTML = ""; });
}
clearStale.seen = {};

// ---------- tiny helpers ----------
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, html) => { const e = document.createElement(tag);
  Object.assign(e, props); if (html != null) e.innerHTML = html; return e; };
const TOAST_ICONS = {
  ok: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>',
  err: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="12.5"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
  warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9 16H3L12 3z"/><line x1="12" y1="10" x2="12" y2="14"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  "": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><line x1="12" y1="7.5" x2="12.01" y2="7.5"/></svg>',
};
function toast(msg, kind = "") {
  const t = $("toast");
  t.className = "toast " + kind;
  t.innerHTML = `<span class="toast-icon">${TOAST_ICONS[kind] || TOAST_ICONS[""]}</span>
    <span class="toast-msg"></span>
    <button class="toast-close" type="button" aria-label="Dismiss">&times;</button>`;
  t.querySelector(".toast-msg").textContent = msg;
  requestAnimationFrame(() => t.classList.add("show"));
  clearTimeout(toast._t);
  const hide = () => t.classList.remove("show");
  t.querySelector(".toast-close").onclick = () => { clearTimeout(toast._t); hide(); };
  toast._t = setTimeout(hide, 5200);
}
function showBusy(msg, sub, withBar) {
  $("busyMsg").textContent = msg || "Working…"; $("busySub").textContent = sub || "";
  $("busyBarWrap").style.display = withBar === false ? "none" : "block";
  setBusyBar(0); $("busy").classList.add("show");
}
function busyMsg(msg, sub) { $("busyMsg").textContent = msg; if (sub != null) $("busySub").textContent = sub; }
function setBusyBar(p) { $("busyBar").style.width = p + "%"; }
function hideBusy() { $("busy").classList.remove("show"); }
// every api reply is json ({error} on failure), but a crash or a proxy can still
// hand back html or nothing, so these never throw into the page
async function api(url, opts) {
  let r;
  try { r = await fetch(url, opts); }
  catch (e) { return { error: "Cannot reach the server. Is python app.py still running?" }; }
  try { return await r.json(); }
  catch (e) { return { error: `Unexpected server response (HTTP ${r.status}). Check the terminal running app.py.` }; }
}
const jget = (url) => api(url);
const jpost = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body) });
const jform = (url, fd) => api(url, { method: "POST", body: fd });
const jdelete = (url) => api(url, { method: "DELETE" });

// column names, filenames and category labels all come out of the user's csv,
// so anything built with innerhtml goes through esc()
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
function esc(v) { return String(v ?? "").replace(/[&<>"']/g, (ch) => ESC[ch]); }
function opt(value, text = value) { return new Option(text, value); }
function pill(t) { return `<span class="pill ${esc(t)}">${esc(t)}</span>`; }
function num(v) { return (v === null || v === undefined || v === "") ? "" : esc(v); }

// ---------- hover-definition glossary (data summary, model training, model
// testing, spline, clinical epi only) ----------
const GLOSSARY = {
  "AUROC": "Area under the ROC curve. How well the model tells apart who will vs won't have the outcome. 0.5 = random guessing, 1.0 = perfect.",
  "AUPRC": "Area under the precision-recall curve. Like AUROC, but more informative when the outcome is rare.",
  "Sensitivity": "Of patients who truly had the outcome, the % the model correctly caught. Also called recall.",
  "Specificity": "Of patients who truly did not have the outcome, the % correctly cleared.",
  "PPV": "Positive predictive value. When the model predicts “yes,” how often it's actually right. Also called precision.",
  "NPV": "Negative predictive value. When the model predicts “no,” how often it's actually right.",
  "F1": "A single score balancing precision and sensitivity.",
  "F2": "Like F1, but weights sensitivity (catching true cases) higher than precision.",
  "Brier": "How well-calibrated the predicted probabilities are. Lower is better.",
  "Accuracy": "The % of all predictions the model got right.",
  "BalancedAccuracy": "Accuracy adjusted so rare classes count as much as common ones.",
  "MacroF1": "F1 averaged evenly across every class, regardless of how common each one is.",
  "R2": "The % of variation in the outcome explained by the model.",
  "RMSE": "Typical prediction error size, in the outcome's own units. Penalizes large misses more than small ones.",
  "MAE": "Typical prediction error size, in the outcome's own units, treating all misses equally.",
  "Threshold": "The predicted-probability cutoff above which a prediction counts as “yes” (default 50%).",
  "N": "Number of rows the statistic was computed on.",
  "SD": "Standard deviation. How spread out the values are around the mean.",
  "IQR": "Interquartile range. The span covering the middle 50% of values.",
  "Grid search": "Automatically trying different model settings (hyperparameters) to find the combination that scores best.",
  "SMOTE": "Synthetic Minority Over-sampling Technique: adds synthetic minority-class rows to the training set. It usually worsens calibration without improving discrimination (van den Goorbergh et al., JAMIA 2022), so it is off by default. If you use it, check calibration.",
  "Confounding variables": "Variables forced into the model and tagged, rather than dropped or left to compete for selection.",
  "Restricted Cubic Spline": "A smooth curve, allowed to bend only at its knots, fit through the data instead of a straight line.",
  "Knots": "The points along the predictor's range where the spline's curve is allowed to change shape.",
  "AIC": "A score for comparing model fit. Lower generally means a better-fitting model.",
  "Odds Ratio": "How much more likely the outcome is, in odds terms, for the exposed group vs. the unexposed group.",
  "Risk Ratio": "How much more likely the outcome is, in plain risk/probability terms, for the exposed group.",
  "NNT": "Number needed to treat. How many patients you'd need to treat to prevent one additional bad outcome.",
  "NNH": "Number needed to harm. How many patients you'd need to expose to cause one additional bad outcome.",
  "Kaplan-Meier": "A curve showing the fraction of patients who have not yet had an event (like death) over time.",
  "Cox proportional hazards": "A model estimating how much various factors speed up or slow down time-to-event, as hazard ratios.",
  "Log-rank": "A statistical test comparing two Kaplan-Meier curves to see if they're really different.",
  "concordance": "Like AUROC, but for survival models. How often the model ranks patients' event risk in the right order.",
  "HR": "Hazard ratio. How much a covariate speeds up (>1) or slows down (<1) time to the event.",
  "Hosmer-Lemeshow": "A test checking whether a model's predicted probabilities match the real observed event rates.",
  "HL statistic": "The Hosmer-Lemeshow test statistic. Larger values suggest worse calibration.",
  "2×2 table": "The four-cell table of exposed/unexposed crossed with outcome yes/no, used for OR/RR/NNT calculations.",
  "Pearson r": "A number from -1 to 1 measuring the strength and direction of a straight-line relationship between a feature and the outcome. Near 0 means little linear relationship, but there could still be a non-linear one.",
  // calibration
  "CalibrationSlope": "Calibration slope. 1 is ideal. Below 1 means predictions are too extreme (high risks too high, low risks too low), the usual sign of overfitting; above 1 means they are too modest.",
  "CalibrationInTheLarge": "Calibration-in-the-large (CITL). 0 is ideal. Below 0 means risks are overestimated on average, above 0 means they are underestimated.",
  "OE_ratio": "Observed to expected ratio: events that happened divided by events the model predicted. 1 is ideal; below 1 means the model predicted too many events.",
  "ICI": "Integrated calibration index: the average gap between predicted risk and observed risk, read off a smoothed calibration curve. 0 is ideal; 0.02 means risks are off by 2 percentage points on average.",
  "E50": "Median gap between predicted and observed risk.",
  "E90": "90th percentile of the gap between predicted and observed risk: 90% of patients have a smaller error than this.",
  // thresholds and clinical usefulness
  "Decision threshold": "The predicted risk above which a patient counts as positive. It is chosen on the validation set and then used unchanged on the test set, so test results stay honest.",
  "Threshold method": "How the threshold was chosen. Youden: best balance of sensitivity and specificity. Target sensitivity or specificity: the cutoff that reaches the target. Outcome rate: the event rate. Fixed: a number you typed.",
  "Youden": "Youden index: sensitivity plus specificity minus 1. The threshold where it is highest gives the best balance of the two.",
  "Recalibration": "Refits the intercept and slope of the predicted risks (on the log-odds scale) on the validation set so they match the observed event rate. Risks change, the ranking of patients does not, so AUROC stays the same.",
  "Decision curve": "Net benefit plotted across threshold probabilities. A model is clinically useful where its curve is above both treat all and treat none.",
  "Net benefit": "True positives minus false positives per patient, with each false positive weighted by how much a false alarm is worth at that threshold. It puts benefit and harm on one scale; higher is better.",
  "Threshold probability": "The risk at which you would act (operate, investigate, monitor). Choosing it says how you weigh a missed case against an unnecessary intervention.",
  // survival and competing risks
  "Median survival": "The time by which half the patients have had the event. Not reached when fewer than half have.",
  "At risk": "Patients still being followed and event free just before that time.",
  "Competing risks": "An event that stops the event of interest from ever happening, such as death before a revision. Treating it as ordinary censoring makes 1 minus Kaplan-Meier overestimate the incidence.",
  "Cumulative incidence": "The chance of having had the event of interest by a given time, allowing for competing events (Aalen-Johansen estimator).",
  "Naive 1 minus KM": "1 minus Kaplan-Meier, treating competing events as censored. It overestimates the incidence when competing events happen; shown only for comparison.",
  "Gray's test": "Tests whether cumulative incidence curves differ between groups when there are competing risks. A small p suggests the groups differ.",
  "Proportional hazards test": "Checks whether a covariate's hazard ratio stays constant over follow-up (Schoenfeld residuals). A small p suggests it changes over time, so a single hazard ratio may not describe it well.",
  // 2x2
  "Risk difference": "Risk in the exposed group minus risk in the unexposed group. One divided by it is the NNT or NNH.",
  "Fisher exact": "An exact p-value for a 2×2 table. Preferred over chi-square when any expected count is below 5.",
  "Chi-square": "Pearson chi-square test of association. For a 2×2 table it uses the Yates continuity correction, as R does by default.",
  // models and sample size
  "Odds ratios": "From a logistic regression of the outcome on all the model's predictors together, fitted on the training set. Each odds ratio is adjusted for the other predictors; numeric predictors are per one unit.",
  "EPP": "Events per parameter: patients in the smaller outcome group divided by the number of model parameters (each numeric predictor counts 1, each extra category level counts 1). Low values mean a higher risk of overfitting.",
  "C statistic": "The AUROC you expect the model to reach, taken from earlier studies of similar models.",
  "Cox-Snell R2": "An R² for logistic models whose maximum is below 1. Riley et al. suggest a conservative 15% of the maximum when nothing is known.",
  "Nagelkerke R2": "Cox-Snell R² rescaled so that its maximum is 1.",
  "Shrinkage": "How much predictions need to be pulled toward the average to correct for overfitting. 0.9 means about 10% shrinkage is expected.",
  // teaching
  "Lockbox": "Patients set aside at the very start and never used for training, tuning or choosing anything. Their score is the honest estimate of how the model would do on new patients.",
  "Optimism": "Reported AUROC minus the honest lockbox AUROC. Positive values mean the approach flatters the model.",
  "Chance margin": "Roughly how far a single lockbox AUROC can move by chance alone (about 2 standard errors, more with few events). Gaps smaller than this may just be noise.",
};
function term(label, key) {
  const tip = GLOSSARY[key || label];
  return tip ? `<span class="term" tabindex="0" data-tip="${esc(tip)}">${esc(label)}</span>` : esc(label);
}

// ---------- numbers, intervals and notices ----------
// null shows as n/a and objects never reach a cell, so no [object Object]
const isNum = (v) => typeof v === "number" && isFinite(v);
function fmt(v, dp) {
  if (v === null || v === undefined || v === "") return "n/a";
  if (isNum(v)) return dp == null ? String(v) : v.toFixed(dp);
  if (typeof v === "object") return "n/a";
  return String(v);
}
function hasCI(ci) { return Array.isArray(ci) && ci.length === 2 && ci[0] != null && ci[1] != null; }
// "0.784 (0.741 to 0.826)", as text for csv and toasts
function fmtCI(est, ci, dp) {
  const e = fmt(est, dp);
  return est != null && hasCI(ci) ? `${e} (${fmt(ci[0], dp)} to ${fmt(ci[1], dp)})` : e;
}
// same text, interval muted, for table cells
function ciHtml(est, ci, dp) {
  const e = esc(fmt(est, dp));
  return est != null && hasCI(ci) ? `${e} <span class="ci">(${esc(fmt(ci[0], dp))} to ${esc(fmt(ci[1], dp))})</span>` : e;
}
function fmtP(p) {
  if (!isNum(p)) return "n/a";
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}
function fmtPct(v, dp = 1) { return isNum(v) ? (v * 100).toFixed(dp) : "n/a"; }
// percent estimate with interval, numbers only ("87.3 (82.0 to 91.2)"), header carries the %
function pctCI(est, lo, hi) {
  if (!isNum(est)) return "n/a";
  return isNum(lo) && isNum(hi) ? `${fmtPct(est)} (${fmtPct(lo)} to ${fmtPct(hi)})` : fmtPct(est);
}
function plural(n, one, many) { return `${n} ${n === 1 ? one : (many || one + "s")}`; }
const NOTICE_ICON = { warn: TOAST_ICONS.warn, err: TOAST_ICONS.err, ok: TOAST_ICONS.ok, info: TOAST_ICONS[""] };
// html must already be escaped by the caller
function notice(kind, html) {
  return `<div class="disclaimer ${kind === "warn" ? "" : kind}"><span class="disc-icon">${NOTICE_ICON[kind] || NOTICE_ICON.info}</span><span class="disc-text">${html}</span></div>`;
}
function noticeList(kind, items, title) {
  const list = (items || []).filter(Boolean);
  if (!list.length) return "";
  if (list.length === 1 && !title) return notice(kind, esc(list[0]));
  return notice(kind, (title ? `<b>${esc(title)}</b>` : "") + `<ul>${list.map(x => `<li>${esc(x)}</li>`).join("")}</ul>`);
}
// numbers typed as "30, 90, 365" -> [30, 90, 365]; null when empty, undefined when bad
function parseNumberList(text) {
  const parts = String(text || "").split(/[\s,;]+/).filter(Boolean);
  if (!parts.length) return null;
  const nums = parts.map(Number);
  return nums.every(isFinite) ? nums : undefined;
}

// ---------- tab navigation ----------
function refreshTab(tab) {
  if (tab === "summary") loadSummary();
  if (tab === "train") initTrain();
  if (tab === "test") loadModels();
  if (tab === "teach") initTeach();
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
function goToSampleSize() {
  document.querySelector('[data-tab="epi"]').click();
  setTimeout(() => $("sampleSizePanel").scrollIntoView({ behavior: "smooth", block: "start" }), 50);
}

// =====================================================================
// 1. upload
// =====================================================================
const drop = $("drop"), fileInput = $("fileInput");
drop.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

// demo data: the server makes the synthetic cohort, then it goes through the normal upload
$("demoBtn").addEventListener("click", async () => {
  const b = $("demoBtn"); b.disabled = true;
  try {
    const r = await fetch("/api/demo.csv");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    handleFile(new File([blob], "demo_hip_knee_synthetic.csv", { type: "text/csv" }));
  } catch (e) {
    toast("Could not load the demo data: " + e.message, "err");
  } finally { b.disabled = false; }
});

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) { toast("Please choose a .csv file.", "err"); return; }
  // instant in-browser preveiw (local, does not touch the server)
  Papa.parse(file, { header: true, preview: 20, skipEmptyLines: true,
    complete: (res) => renderPreview(res.meta.fields, res.data), error: () => {} });

  // send the raw file to the local python proccess (localhost) for real parsing
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
    if (p >= 100) busyMsg("Reading your file…", "detecting column types (stays on your machine)");
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
    State.columns = d.columns; State.confirmed = false; dataChanged();
    $("imbalanceNote").style.display = "none"; $("confirmMsg").textContent = "";
    $("uploadInfo").innerHTML = `<b>${esc(d.filename)}</b>: ${d.n_rows} rows × ${d.n_cols} columns`;
    $("sessionTag").innerHTML = `Dataset: <b style="color:#fff">${esc(d.filename)}</b><br>${d.n_rows} rows`;
    renderTypeTable(d.columns); buildStratOptions(d.columns);
    resetCohort();
    $("typePanel").style.display = $("splitPanel").style.display = $("cohortPanel").style.display = "block";
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
  t.appendChild(el("tr", {}, fields.map(f => `<th>${esc(f)}</th>`).join("")));
  rows.forEach(r => t.appendChild(el("tr", {}, fields.map(f => `<td>${num(r[f])}</td>`).join(""))));
}

const TYPES = ["binary", "categorical", "continuous", "date", "exclude"];
// never defaults to zero: 0 is a real value (albumin 0 g/dL), so zero-fill is
// only ever an explicit choice for counts where missing means none
function defaultMissingDecision(pct) {
  if (pct > 50) return "remove";
  return "impute"; // 0 < pct <= 50, fill value learned on the training set only
}
const HIGH_MISSING_NOTE = 30;
function missingSelectHtml(c, i) {
  if (!(c.pct_missing > 0)) return `<span class="muted small">none</span>`;
  const def = defaultMissingDecision(c.pct_missing);
  const opt = (v, label) => `<option value="${v}"${def === v ? " selected" : ""}>${label}</option>`;
  const note = c.pct_missing >= HIGH_MISSING_NOTE
    ? `<div class="small" style="color:var(--warn);margin-top:4px">High missingness: consider whether missing is informative (impute + indicator keeps that signal).</div>`
    : "";
  return `<select id="miss_${i}" class="small">
      ${opt("impute", "Impute (median / most frequent, from training set)")}
      ${opt("impute_indicator", "Impute + missing indicator")}
      ${opt("include", "Leave missing (model imputes in training)")}
      ${opt("zero", "Fill 0 (counts where missing means none)")}
      ${opt("remove", "Remove column")}
    </select>${note}`;
}
function renderTypeTable(cols) {
  const t = $("typeTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, `<th>Column</th><th>Detected</th><th>Unique</th><th>% missing</th>
    <th>Type override</th><th>Missing-data handling</th>`));
  // ids use the column's position, a header can contain quotes or brackets
  cols.forEach((c, i) => {
    const tr = el("tr");
    const radios = TYPES.map(tp => `<label class="small" style="font-weight:500;margin-right:8px">
        <input type="radio" name="ty_${i}" value="${tp}" ${c.type === tp ? "checked" : ""}> ${tp}</label>`).join("");
    tr.innerHTML = `<td><b>${esc(c.name)}</b></td><td>${pill(c.type)}</td><td>${c.n_unique}</td>
      <td ${c.high_missing ? 'style="color:var(--warn);font-weight:600"' : ""}>${c.pct_missing}%</td>
      <td>${radios}</td><td>${missingSelectHtml(c, i)}</td>`;
    t.appendChild(tr);
  });
}

function buildStratOptions(cols) {
  // a new upload keeps the stratify column when the new file has it too
  const sel = $("stratCol"), prev = sel.value;
  sel.innerHTML = `<option value="">None</option>`;
  cols.filter(c => c.type === "binary" || c.type === "categorical")
    .forEach(c => sel.appendChild(opt(c.name)));
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  else if ($("splitMethod").value === "stratified") $("splitMethod").value = "random";
  $("smoteBox").style.display = sel.value ? "block" : "none";
}
$("stratCol").addEventListener("change", () => {
  $("smoteBox").style.display = $("stratCol").value ? "block" : "none";
});
const IMBALANCE_TEXT = "Resampling such as SMOTE usually worsens calibration without improving " +
  "discrimination (van den Goorbergh et al., JAMIA 2022), so the default is no resampling and the " +
  "decision threshold is chosen instead. If you do use SMOTE, check calibration and consider recalibration.";
$("splitMethod").addEventListener("change", e => {
  if (e.target.value === "stratified" && !$("stratCol").value)
    toast("Choose a column to stratify by.", "");
});

$("confirmBtn").addEventListener("click", async () => {
  const types = {}, missing = {};
  State.columns.forEach((c, i) => {
    const r = document.querySelector(`input[name="ty_${i}"]:checked`);
    types[c.name] = r ? r.value : c.type;
    const m = $(`miss_${i}`); if (m) missing[c.name] = m.value;
  });
  const ratios = [+$("rTrain").value, +$("rVal").value, +$("rTest").value];
  if (ratios.some(x => !(x > 0))) { toast("Train, validation and test % must all be above zero.", "err"); return; }
  if ($("splitMethod").value === "stratified" && !$("stratCol").value) {
    toast("Choose a column to stratify by, or switch the method to Random.", "err"); return;
  }
  const cohort = cohortPayload();
  $("confirmMsg").textContent = "Splitting & saving…";
  $("cohortProblems").innerHTML = "";
  showBusy("Cleaning & splitting dataset…", cohort.length
    ? `applying ${plural(cohort.length, "cohort filter")}, then saving train / validation / test partitions`
    : "saving train / validation / test partitions", false);
  const body = {
    types, missing, ratios, method: $("splitMethod").value,
    stratify_col: $("stratCol").value || null, smote: $("smoteToggle").checked
  };
  if (cohort.length) body.cohort = cohort;
  const d = await jpost("/api/confirm", body);
  hideBusy();
  if (d.error) {
    toast(d.error, "err"); $("confirmMsg").textContent = d.problems || d.cohort ? "Fix the cohort filters above." : d.error;
    if (d.problems || d.cohort) {
      showCohortProblems(d);
      if (d.cohort) renderCohortFlow($("cohortResult"), d.cohort, {});
      $("cohortPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    return;
  }
  State.confirmed = true; State.coltypes = types; dataChanged();
  document.querySelector('[data-tab="upload"]').classList.add("done");
  $("confirmMsg").innerHTML = `<span class="ok-text">Locked.</span> train ${d.n_train} · val ${d.n_val} · test ${d.n_test} (${esc(d.method)})`;
  const nFilled = Object.keys(d.fill_values || {}).length;
  if (nFilled) $("confirmMsg").innerHTML += ` · fill values for ${nFilled} column${nFilled === 1 ? "" : "s"} learned on the training set`;
  State.cohort = d.cohort || null;
  if (State.cohort) {
    $("confirmMsg").innerHTML += ` · cohort ${fmtInt(State.cohort.n_start)} to ${fmtInt(State.cohort.n_final)} rows`;
    // the confirm reply may not carry the diagram; reuse the preview's when the filters match
    if (!State.cohort.png && COHORT.preview && COHORT.previewKey === JSON.stringify(cohort))
      State.cohort.png = COHORT.preview.png;
    renderCohortFlow($("cohortResult"), State.cohort, { confirmed: true });
  }
  if (d.smote_hint && d.smote_hint.imbalanced) {
    const pct = (d.smote_hint.minority_fraction * 100).toFixed(1);
    $("imbalanceNote").innerHTML = `<b>Class imbalance:</b> the minority class of ${esc(d.smote_hint.column)} is ${pct}%. ` +
      IMBALANCE_TEXT;
    $("imbalanceNote").style.display = "block";
    toast(`Minority class ${pct}%. No resampling by default; choose the decision threshold instead.`, "warn");
  } else {
    $("imbalanceNote").style.display = "none";
  }
  toast("Dataset confirmed and saved.", "ok");
  document.querySelector('[data-tab="summary"]').click();
});

// =====================================================================
// 1b. cohort builder (optional, runs on the raw upload before the split)
// =====================================================================
const COHORT_OPS = [
  ["in", "Keep: is one of these codes"],
  ["not_in", "Drop: is one of these codes"],
  ["between", "Keep: between two numbers"],
  ["gte", "Keep: at least"],
  ["lte", "Keep: at most"],
  ["eq", "Keep: equals"],
  ["neq", "Drop: equals"],
  ["not_missing", "Keep: has a value"],
];
const COHORT_OP_HINT = {
  in: "Keeps a row when any ticked column holds one of the codes. Blank cells never match.",
  not_in: "Drops a row when any ticked column holds one of the codes. Rows with blank cells stay.",
  between: "Keeps values from low to high, both included. One column only; missing or non-numeric values are dropped.",
  gte: "Keeps values at or above the number. One column only; missing values are dropped.",
  lte: "Keeps values at or below the number. One column only; missing values are dropped.",
  eq: "Keeps rows equal to the value (text ignores case). One column only; missing values are dropped.",
  neq: "Drops rows equal to the value; everything else, blanks included, stays. One column only.",
  not_missing: "Keeps rows that have a value in every ticked column.",
};
const COHORT_MULTI = new Set(["in", "not_in", "not_missing"]);
const COHORT = { filters: [], nextId: 1, preview: null, previewKey: null };
const fmtInt = (n) => isNum(n) ? n.toLocaleString() : fmt(n);

function resetCohort() {
  COHORT.filters = []; COHORT.preview = null; COHORT.previewKey = null; State.cohort = null;
  $("cohortProblems").innerHTML = ""; $("cohortResult").innerHTML = "";
  renderCohortFilters();
}
function addCohortFilter() {
  COHORT.filters.push({ id: COHORT.nextId++, columns: [], op: "in", codes: "", lo: "", hi: "", value: "", label: "", search: "" });
  renderCohortFilters();
  const last = $("cohortFilters").lastElementChild;
  if (last) last.querySelector(".colpick-tools input").focus();
}
// filter state -> request body; values are sent raw, the server parses and checks them
function cohortPayload() {
  return COHORT.filters.map(f => {
    const o = { columns: f.columns.slice(), op: f.op };
    if (f.op === "in" || f.op === "not_in") o.values = f.codes;
    else if (f.op === "between") o.values = [f.lo.trim(), f.hi.trim()];
    else if (f.op !== "not_missing") o.values = [f.value.trim()];
    if (f.label.trim()) o.label = f.label.trim();
    return o;
  });
}
function cohortTitle(f) {
  const cols = f.columns.length ? (f.columns.length > 3 ? `${f.columns.slice(0, 2).join(", ")} and ${f.columns.length - 2} more` : f.columns.join(", ")) : "no column yet";
  return f.label.trim() || `${COHORT_OPS.find(o => o[0] === f.op)[1]} (${cols})`;
}
function cohortValueHtml(f) {
  if (f.op === "in" || f.op === "not_in")
    return `<label>Codes</label><textarea data-k="codes" placeholder="27130, 27447&#10;or a range: 27130-27132&#10;or a wildcard: M16*">${esc(f.codes)}</textarea>`;
  if (f.op === "between")
    return `<label>From and to (both included)</label><div class="pair">
      <input type="number" step="any" data-k="lo" placeholder="low" value="${esc(f.lo)}"><span>to</span>
      <input type="number" step="any" data-k="hi" placeholder="high" value="${esc(f.hi)}"></div>`;
  if (f.op === "gte" || f.op === "lte")
    return `<label>${f.op === "gte" ? "At least" : "At most"}</label><input type="number" step="any" data-k="value" placeholder="e.g. 18" value="${esc(f.value)}">`;
  if (f.op === "eq" || f.op === "neq")
    return `<label>Value</label><input type="text" data-k="value" placeholder="e.g. F or 1" value="${esc(f.value)}">`;
  return `<label>Value</label><div class="small muted" style="padding-top:6px">No value needed.</div>`;
}
function renderCohortFilters() {
  const box = $("cohortFilters"); box.innerHTML = "";
  const n = COHORT.filters.length;
  $("cohortPreviewBtn").style.display = n ? "" : "none";
  $("cohortClearBtn").style.display = n ? "" : "none";
  if (!n) {
    box.innerHTML = `<div class="cf-empty">No filters: every row goes into the split. Add a filter to restrict the cohort.</div>`;
    cohortChanged(); return;
  }
  const cols = State.columns || [];
  COHORT.filters.forEach((f, i) => {
    const card = el("div", { className: "cf" });
    card.dataset.id = f.id;
    const opOpts = COHORT_OPS.map(([v, l]) => `<option value="${v}"${v === f.op ? " selected" : ""}>${esc(l)}</option>`).join("");
    const colBoxes = cols.map(c => `<label title="${esc(c.name)} (${esc(c.type)})"><input type="checkbox" value="${esc(c.name)}"${f.columns.includes(c.name) ? " checked" : ""}> ${esc(c.name)}</label>`).join("");
    card.innerHTML = `
      <div class="cf-head"><span class="acc-num">${i + 1}</span><span class="cf-title"></span>
        <button class="ghost sm" type="button" data-act="up" title="Move up" aria-label="Move up"${i === 0 ? " disabled" : ""}>&#9650;</button>
        <button class="ghost sm" type="button" data-act="down" title="Move down" aria-label="Move down"${i === n - 1 ? " disabled" : ""}>&#9660;</button>
        <button class="ghost sm" type="button" data-act="remove" title="Remove this filter" aria-label="Remove">&times;</button></div>
      <div class="cf-grid">
        <div class="cf-cols"><label>Column(s)</label>
          <div class="colpick-tools"><input type="text" placeholder="Search columns" value="${esc(f.search)}">
            <button class="ghost sm" type="button" data-act="tick" title="Tick every column shown">Tick shown</button>
            <button class="ghost sm" type="button" data-act="untick" title="Untick all">Clear</button></div>
          <div class="checklist varpick">${colBoxes}</div>
          <div class="cf-picked"></div></div>
        <div><label>Rule</label><select data-k="op">${opOpts}</select>
          <div class="hint cf-ophint"></div>
          <label style="margin-top:10px">Label for the flow diagram (optional)</label>
          <input type="text" data-k="label" placeholder="e.g. Primary total hip arthroplasty" value="${esc(f.label)}"></div>
        <div class="cf-val">${cohortValueHtml(f)}</div>
      </div>`;
    box.appendChild(card);
    syncCohortCard(card, f);
    applyColSearch(card, f.search);
  });
  cohortChanged();
}
function syncCohortCard(card, f) {
  card.querySelector(".cf-title").textContent = cohortTitle(f);
  card.querySelector(".cf-ophint").textContent = COHORT_OP_HINT[f.op] || "";
  const picked = card.querySelector(".cf-picked");
  const many = f.columns.length > 1 && !COHORT_MULTI.has(f.op);
  picked.innerHTML = many
    ? `<span style="color:var(--warn);font-weight:600">This rule works on one column; ${f.columns.length} are ticked.</span>`
    : f.columns.length ? `${plural(f.columns.length, "column")} ticked` : "Tick at least one column.";
}
function applyColSearch(card, q) {
  const s = q.trim().toLowerCase();
  card.querySelectorAll(".varpick label").forEach(l => {
    l.style.display = !s || l.textContent.toLowerCase().includes(s) ? "" : "none";
  });
}
function cohortById(card) { return COHORT.filters.find(f => f.id === +card.dataset.id); }
$("cohortFilters").addEventListener("input", (e) => {
  const card = e.target.closest(".cf"); if (!card) return;
  const f = cohortById(card); if (!f) return;
  if (e.target.closest(".colpick-tools") && e.target.tagName === "INPUT") {
    f.search = e.target.value; applyColSearch(card, f.search); return;
  }
  const k = e.target.dataset.k;
  if (k && k !== "op") { f[k] = e.target.value; syncCohortCard(card, f); cohortChanged(); }
});
$("cohortFilters").addEventListener("change", (e) => {
  const card = e.target.closest(".cf"); if (!card) return;
  const f = cohortById(card); if (!f) return;
  if (e.target.type === "checkbox") {
    f.columns = [...card.querySelectorAll(".varpick input:checked")].map(c => c.value);
    syncCohortCard(card, f); cohortChanged();
  } else if (e.target.dataset.k === "op") {
    f.op = e.target.value;
    card.querySelector(".cf-val").innerHTML = cohortValueHtml(f);
    syncCohortCard(card, f); cohortChanged();
  }
});
$("cohortFilters").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]"); if (!b) return;
  const card = b.closest(".cf"), f = cohortById(card); if (!f) return;
  const i = COHORT.filters.indexOf(f), act = b.dataset.act;
  if (act === "tick" || act === "untick") {
    card.querySelectorAll(".varpick label").forEach(l => {
      const cb = l.querySelector("input");
      if (act === "untick") cb.checked = false;
      else if (l.style.display !== "none") cb.checked = true;
    });
    f.columns = [...card.querySelectorAll(".varpick input:checked")].map(c => c.value);
    syncCohortCard(card, f); cohortChanged(); return;
  }
  if (act === "remove") COHORT.filters.splice(i, 1);
  if (act === "up" && i > 0) COHORT.filters.splice(i - 1, 0, COHORT.filters.splice(i, 1)[0]);
  if (act === "down" && i < COHORT.filters.length - 1) COHORT.filters.splice(i + 1, 0, COHORT.filters.splice(i, 1)[0]);
  renderCohortFilters();
});
$("cohortAddBtn").addEventListener("click", addCohortFilter);
$("cohortClearBtn").addEventListener("click", () => { resetCohort(); toast("Cohort filters removed.", ""); });
$("cohortPreviewBtn").addEventListener("click", previewCohort);

// marks an old preview as out of date once the filters change
function cohortChanged() {
  const n = COHORT.filters.length;
  $("cohortStatus").textContent = n ? `${plural(n, "filter")}, applied in this order` : "";
  const stale = COHORT.preview && COHORT.previewKey !== JSON.stringify(cohortPayload());
  const flag = $("cohortResult").querySelector(".cohort-stale");
  if (flag) flag.style.display = stale ? "" : "none";
}
function showCohortProblems(d) {
  const items = d.problems && d.problems.length ? d.problems : [d.error || "The cohort filters could not be applied."];
  $("cohortProblems").innerHTML = `<div class="notices">${noticeList("err", items, "Fix these filters first:")}</div>`;
}
async function previewCohort() {
  const filters = cohortPayload();
  if (!filters.length) { toast("Add a filter first.", "err"); return; }
  $("cohortProblems").innerHTML = "";
  const btn = $("cohortPreviewBtn"); btn.disabled = true; btn.textContent = "Counting…";
  let d;
  try { d = await jpost("/api/cohort/preview", { filters }); }
  finally { btn.disabled = false; btn.textContent = "Preview cohort"; }
  if (d.error || d.problems) {
    showCohortProblems(d); toast("The cohort filters need fixing.", "err"); return;
  }
  COHORT.preview = d; COHORT.previewKey = JSON.stringify(filters);
  renderCohortFlow($("cohortResult"), d, { preview: true });
  if (d.n_final === 0) toast("No rows are left after these filters.", "warn");
}
// flow = {flow, lines, n_start, n_final, png}; used for the preview, after
// confirm, and on the data summary tab
function renderCohortFlow(box, d, { preview = false, confirmed = false, idx = "" } = {}) {
  const flow = d.flow || [];
  const removed = isNum(d.n_start) && isNum(d.n_final) ? d.n_start - d.n_final : null;
  const tid = `cohortFlowTable${idx}`;
  const anyExtra = flow.some(s => s.n_missing || s.n_non_numeric || s.n_top_coded);
  const rows = flow.map(s => {
    const kept = !["not_in", "neq"].includes(s.op);
    const extra = [];
    if (s.n_missing) extra.push(`${fmtInt(s.n_missing)} missing`);
    if (s.n_non_numeric) extra.push(`${fmtInt(s.n_non_numeric)} not numeric`);
    if (s.n_top_coded) extra.push(`${fmtInt(s.n_top_coded)} top-coded`);
    return `<tr><td>${esc(s.step)}</td><td>${esc(s.label)} <span class="tag ${kept ? "ok" : "warn"}">${kept ? "keep" : "drop"}</span></td>
      <td class="nw">${fmtInt(s.n_before)}</td><td class="nw">${fmtInt(s.n_removed)}</td><td class="nw"><b>${fmtInt(s.n_after)}</b></td>
      ${anyExtra ? `<td class="small">${esc(extra.join(", "))}</td>` : ""}</tr>`;
  }).join("");
  const lineList = d.lines && d.lines.length ? d.lines : cohortLines(d);
  const lines = lineList.map(l => `<li>${esc(l)}</li>`).join("");
  box.innerHTML = `
    ${preview ? `<div class="cohort-stale" style="display:none">${notice("warn", "Filters changed since this preview. Preview again to update the counts.")}</div>` : ""}
    ${confirmed ? `<div class="notices">${notice("ok", "<b>Cohort applied.</b> These rows went into the train / validation / test split.")}</div>` : ""}
    <div class="cards" style="margin-top:14px">
      <div class="card"><div class="lbl">Rows in file</div><div class="big">${fmtInt(d.n_start)}</div></div>
      <div class="card"><div class="lbl">Removed</div><div class="big">${removed == null ? "n/a" : fmtInt(removed)}</div></div>
      <div class="card"><div class="lbl">In cohort</div><div class="big">${fmtInt(d.n_final)}</div></div>
    </div>
    <div class="flow-wrap">
      <div>
        <div class="tbl-tools"><h4>Patient flow</h4><span class="flow-btns"></span></div>
        <ol class="flowlines">${lines}</ol>
        <div class="tbl-tools"><h4>Step by step</h4><span class="tbl-btns"></span></div>
        <div class="scrollx"><table id="${tid}"><tr><th>Step</th><th>Filter</th><th>Rows before</th><th>Removed</th><th>Rows after</th>${anyExtra ? "<th>Of the removed</th>" : ""}</tr>${rows}</table></div>
      </div>
      ${d.png ? `<div class="flow-png" style="flex:0 1 auto;min-width:260px"><img class="plot" src="${d.png}" style="max-width:440px"><div class="flex png-btns"></div></div>` : ""}
    </div>`;
  box.querySelector(".flow-btns").appendChild(dlButton("Copy text", () => {
    const text = lineList.join("\n");
    navigator.clipboard.writeText(text).then(() => toast("Flow copied.", "ok"), () => toast("Could not copy.", "err"));
  }));
  box.querySelector(".tbl-btns").appendChild(dlButton("CSV", () => dlTable(tid, "cohort_flow.csv")));
  if (d.png) {
    const pb = box.querySelector(".png-btns");
    pb.appendChild(dlButton("PNG", () => dlPng(d.png, "cohort_flow.png")));
    pb.appendChild(dlButton("PDF", () => dlPdf("flow", { flow, n_start: d.n_start }, "cohort_flow")));
  }
}
function cohortLines(d) {
  const out = [`Records in file: ${fmtInt(d.n_start)}`];
  (d.flow || []).forEach(s => out.push(s.kind === "include"
    ? `${s.label}: ${fmtInt(s.n_after)}`
    : `${s.label}: ${fmtInt(s.n_removed)} removed, ${fmtInt(s.n_after)} remaining`));
  out.push(`Final cohort: ${fmtInt(d.n_final)}`);
  return out;
}

// =====================================================================
// 2. summary
// =====================================================================
async function loadSummary() {
  const d = await jget("/api/summary");
  if (d.error) { $("summaryEmpty").style.display = "block"; $("summaryBody").style.display = "none";
    $("summaryEmpty").textContent = d.error; return; }
  $("summaryEmpty").style.display = "none"; $("summaryBody").style.display = "block";
  const coh = d.cohort || State.cohort;
  $("cohortSummaryPanel").style.display = coh && coh.flow ? "block" : "none";
  if (coh && coh.flow) renderCohortFlow($("cohortSummary"), coh, { idx: "S" });
  const c = d.card;
  const cards = [["Total N", c.total_n], ["Columns", c.n_columns],
    ["Complete cases", `${c.complete_cases} (${c.pct_complete_cases}%)`]];
  if (c.date_range) cards.push(["Date range", `${c.date_range.start} → ${c.date_range.end}`]);
  $("summaryCards").innerHTML = cards.map(([l, v]) =>
    `<div class="card"><div class="lbl">${l}</div><div class="big">${v}</div></div>`).join("");

  const ct = $("contTable"); ct.innerHTML = "";
  ct.appendChild(el("tr", {}, `<th>Variable</th><th>${term("N")}</th><th>% missing</th><th>Mean</th>
    <th>Median</th><th>${term("SD")}</th><th>${term("IQR")}</th><th>Min</th><th>Max</th>`));
  d.continuous.forEach(r => ct.appendChild(el("tr", {}, `<td>${esc(r.variable)}</td><td>${r.n}</td>
    <td>${r.pct_missing}</td><td>${num(r.mean)}</td><td>${num(r.median)}</td><td>${num(r.sd)}</td>
    <td>${num(r.iqr)}</td><td>${num(r.min)}</td><td>${num(r.max)}</td>`)));
  if (!d.continuous.length) ct.appendChild(el("tr", {}, `<td colspan="9" class="muted">None</td>`));

  const kt = $("catTable"); kt.innerHTML = "";
  kt.appendChild(el("tr", {}, `<th>Variable</th><th>Type</th><th>N</th><th>% missing</th><th>Category</th><th>Count</th><th>%</th>`));
  d.categorical.forEach(v => {
    v.categories.forEach((cat, i) => kt.appendChild(el("tr", {}, `
      ${i === 0 ? `<td rowspan="${v.categories.length}"><b>${esc(v.variable)}</b></td>
        <td rowspan="${v.categories.length}">${pill(v.type)}</td>
        <td rowspan="${v.categories.length}">${v.n}</td>
        <td rowspan="${v.categories.length}">${v.pct_missing}</td>` : ""}
      <td>${esc(cat.category)}</td><td>${cat.count}</td><td>${cat.pct}</td>`)));
  });
  if (!d.categorical.length) kt.appendChild(el("tr", {}, `<td colspan="7" class="muted">None</td>`));
}

// =====================================================================
// 3. train
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
  d.outcomes.forEach(c => o.appendChild(opt(c.name, `${c.name} (${c.type})`)));
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
  const d = await jform("/api/train/upload-split", fd);
  hideBusy();
  if (d.error) {
    toast(d.error, "err");
    $("customSplitMsg").innerHTML = `<span class="err-text">${esc(d.error)}</span>`;
    return;
  }
  $("customSplitMsg").innerHTML = `<span class="ok-text">Using custom split.</span> train ${d.n_train} · val ${d.n_val}`;
  toast("Custom train/validation split applied.", "ok");
  initTrain();
}
function fillMulti(sel, names) { sel.innerHTML = ""; names.forEach(n => sel.appendChild(opt(n))); }
function fillChecklist(box, names) {
  box.innerHTML = "";
  names.forEach(n => {
    const label = el("label"), cb = el("input", { type: "checkbox", value: n });
    label.append(cb, " " + n);
    box.appendChild(label);
  });
}
function multiValues(box) { return [...box.querySelectorAll("input:checked")].map(c => c.value); }

function taskForOutcome(name) {
  const c = TRAIN_META.outcomes.find(x => x.name === name);
  if (!c) return "binary";
  if (c.task) return c.task;                 // the server's own call, same rule as /api/train
  if (c.type === "continuous") return "continuous";
  if (c.type === "binary") return "binary";
  return "multiclass";
}
function onOutcomeChange() {
  const out = $("outcomeSel").value, task = taskForOutcome(out);
  $("taskHint").innerHTML = `Detected task: <b>${task}</b>`;
  $("binaryOpts").style.display = task === "binary" ? "block" : "none";
  onThrMethodChange();
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
    ml.appendChild(el("label", {}, `<input type="checkbox" class="mdl" value="${esc(m)}" checked> ${esc(m)}`));
    const id = "grid_" + m;
    ge.appendChild(el("div", {}, `<label class="small" style="margin-top:6px">${esc(m)}</label>
      <textarea id="${esc(id)}" rows="2">${esc(JSON.stringify(grids[m]))}</textarea>`));
  });
}
function toggleModels(on) { document.querySelectorAll(".mdl").forEach(c => c.checked = on); }

// decision threshold: how each binary model picks its cutoff on the validation set
const THR_METHODS = {
  youden: { label: "Youden", hint: "Picks the cutoff on the validation set where sensitivity plus specificity is highest. The test set never influences it." },
  sensitivity: { label: "target sensitivity", target: "Target sensitivity (0 to 1)", def: 0.9,
    hint: "Picks the highest cutoff that still catches at least this share of validation patients with the outcome." },
  specificity: { label: "target specificity", target: "Target specificity (0 to 1)", def: 0.9,
    hint: "Picks the lowest cutoff that still clears at least this share of validation patients without the outcome." },
  prevalence: { label: "outcome rate", hint: "Uses the outcome rate in the validation set as the cutoff, so a patient is positive when their risk is above average." },
  fixed: { label: "fixed", target: "Threshold (0 to 1)", def: 0.5, hint: "Uses the number you type for every model." },
};
function thrMethodLabel(m, modelName) {
  if (m === "user") return "typed in";
  if (m === "saved") {
    const how = modelName && MODEL_META[modelName] && MODEL_META[modelName].threshold_method;
    return how ? `saved, ${thrMethodLabel(how)}` : "saved";
  }
  return (THR_METHODS[m] && THR_METHODS[m].label) || m || "";
}
function onThrMethodChange() {
  const m = THR_METHODS[$("thrMethod").value] || THR_METHODS.youden;
  $("thrMethodHint").textContent = m.hint;
  $("thrTargetBox").style.display = m.target ? "" : "none";
  if (m.target) {
    $("thrTargetLbl").textContent = m.target;
    $("thrTarget").placeholder = String(m.def);
    if ($("thrTarget").value === "" || $("thrTarget").dataset.auto === "1") {
      $("thrTarget").value = m.def; $("thrTarget").dataset.auto = "1";
    }
  }
}
$("thrMethod").addEventListener("change", onThrMethodChange);
$("thrTarget").addEventListener("input", () => { $("thrTarget").dataset.auto = "0"; });
function syncRecalHint() {
  $("recalSmoteHint").style.display = $("smoteTrain").checked && !$("recalToggle").checked ? "block" : "none";
}
$("smoteTrain").addEventListener("change", syncRecalHint);
$("recalToggle").addEventListener("change", syncRecalHint);

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
    $("confirmModalMsg").textContent = message;
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

// the internal test partition is locked to the models of its first use;
// testing another model on it needs a reason, which goes into the log
function askUnlockReason(newModels) {
  return new Promise(resolve => {
    const modal = $("unlockModal");
    const names = (newModels || []).map(modelLabel);
    $("unlockModels").textContent = names.length
      ? `Not in the first test of this partition: ${names.join(", ")}.` : "";
    $("unlockReason").value = "";
    modal.classList.add("show");
    $("unlockReason").focus();
    const finish = (v) => {
      modal.classList.remove("show");
      $("unlockOk").removeEventListener("click", onOk);
      $("unlockCancel").removeEventListener("click", onCancel);
      modal.removeEventListener("click", onBackdrop);
      resolve(v);
    };
    const onOk = () => {
      const reason = $("unlockReason").value.trim();
      if (!reason) { toast("Give a reason, or cancel.", "err"); $("unlockReason").focus(); return; }
      finish(reason);
    };
    const onCancel = () => finish(null);
    const onBackdrop = (e) => { if (e.target === modal) finish(null); };
    $("unlockOk").addEventListener("click", onOk);
    $("unlockCancel").addEventListener("click", onCancel);
    modal.addEventListener("click", onBackdrop);
  });
}

let trainES = null;
function setTrainingActive(active) {
  // cancel replaces start in the same slot, rather than sitting next to a
  // disabled start button and crowding the progress bar
  $("trainBtn").style.display = active ? "none" : "";
  $("trainBtn").disabled = false;
  $("cancelTrainBtn").style.display = active ? "" : "none";
  $("trainBarWrap").classList.toggle("running", active);
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
  for (const m of models) {
    const t = $("grid_" + m); if (!t) continue;
    try { grids[m] = JSON.parse(t.value); }
    catch (e) { toast(`The ${m} hyperparameter grid isn't valid JSON.`, "err"); return; }
  }
  const cvRaw = $("cvFolds").value.trim(), cvFolds = cvRaw === "" ? null : parseInt(cvRaw, 10);
  if (cvFolds !== null && !(cvFolds >= 2)) { toast("CV folds must be 2 or more (or empty).", "err"); return; }
  const binary = taskForOutcome(outcome) === "binary";
  const thrMethod = $("thrMethod").value, thrInfo = THR_METHODS[thrMethod];
  let thrTarget = null;
  if (binary && thrInfo.target) {
    thrTarget = parseFloat($("thrTarget").value);
    const ok = thrMethod === "fixed" ? thrTarget > 0 && thrTarget < 1 : thrTarget > 0 && thrTarget <= 1;
    if (!ok) { toast(`${thrInfo.target.replace(" (0 to 1)", "")} must be a number between 0 and 1.`, "err"); $("thrTarget").focus(); return; }
  }

  $("trainError").innerHTML = "";
  setTrainingActive(true);
  $("trainLog").style.display = "block"; $("trainLog").textContent = "";
  $("trainResults").style.display = "none"; setBar(0);

  const body = {
    outcome, exclude, confounders, models, grids,
    scoring: $("scoreSel").value, grid_search: $("gridToggle").checked,
    cv_folds: cvFolds, smote: $("smoteTrain").checked,
    threshold: binary && thrMethod === "fixed" ? thrTarget : 0.5
  };
  if (binary) Object.assign(body, { threshold_method: thrMethod, threshold_target: thrTarget,
    recalibrate: $("recalToggle").checked });
  const cfg = await jpost("/api/train", body);
  if (cfg.error) {
    toast("Training could not start.", "err"); setTrainingActive(false);
    $("trainLog").style.display = "none";
    $("trainError").innerHTML = `<div class="notices">${notice("err", `<b>Training could not start.</b> ${esc(cfg.error)}`)}</div>`;
    $("trainError").scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  const metricsRows = [], impWrap = $("importanceWrap"); impWrap.innerHTML = "";
  const orWrap = $("oddsRatiosWrap"); orWrap.innerHTML = "";
  const trainNotes = new Map();     // note -> models it applies to
  const es = new EventSource("/api/train/stream");
  trainES = es;
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "start") log(`▶ Training ${m.n_models} model(s) · ${m.features} features · task=${m.task}`);
    else if (m.event === "progress") { log(`  … ${m.message}`); setBar(m.pct); }
    else if (m.event === "model_error") {
      log(`  ✗ ${m.model}: ${m.error}`);
      metricsRows.push({ name: m.model, error: `Not trained: ${m.error}` });
    }
    else if (m.event === "model_done") {
      log(`  ✓ ${m.model} saved as ${m.saved_as}`); setBar(m.pct);
      metricsRows.push({ name: m.model, metrics: m.metrics || {}, threshold: m.threshold,
        threshold_method: m.threshold_method, recalibrated: m.recalibrated });
      if (m.odds_ratios) orWrap.appendChild(oddsRatioBlock(m.model, m.odds_ratios, m.saved_as || m.model));
      (m.notes || []).forEach(x => { if (!trainNotes.has(x)) trainNotes.set(x, []); trainNotes.get(x).push(m.model); });
      if (m.importance_png) {
        const box = el("div", { className: "plotwrap" },
          `<img class="plot" src="${m.importance_png}" style="max-width:430px"><div class="flex"></div>`);
        const btns = box.querySelector(".flex");
        btns.appendChild(dlButton("PNG", () => dlPng(m.importance_png, `${m.model}_importance.png`)));
        btns.appendChild(dlButton("CSV", () => dlImpCsv(m.importance, `${m.model}_importance.csv`)));
        impWrap.appendChild(box);
      }
    } else if (m.event === "complete") {
      es.close(); trainES = null;
      setTrainingActive(false); setBar(100);
      const nOk = m.results.length, nTotal = models.length;
      if (nOk > 0) {
        renderMetricsTable($("valMetricsTable"), metricsRows);
        // a note every model shares is shown once, otherwise it names its models
        const nOkModels = metricsRows.filter(r => !r.error).length;
        $("valNotes").innerHTML = trainNotes.size ? `<div class="notices">${[...trainNotes].map(([x, ms]) =>
          notice("info", (ms.length < nOkModels ? `<b>${esc(ms.join(", "))}:</b> ` : "") + esc(x))).join("")}</div>` : "";
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
  es.onerror = () => {
    es.close(); trainES = null;
    setTrainingActive(false);
    toast("Training stream error.", "err");
  };
}
function cancelTraining() {
  if (!trainES) return;
  trainES.close();
  trainES = null;
  log("✕ Cancelled. A model already mid-fit on the server may still finish and save on its own; " +
      "no further models in this run will start.");
  toast("Training cancelled.", "");
  setTrainingActive(false);
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
// metrics tables (validation and test): estimates with their 95% CI from
// metrics.CI, the threshold with how it was chosen, never a raw object
const METRIC_LABELS = { CalibrationSlope: "Calibration slope", CalibrationInTheLarge: "CITL",
  OE_ratio: "O:E ratio", BalancedAccuracy: "Balanced accuracy", MacroF1: "Macro F1" };
function metricKeysOf(rows) {
  const keys = [];
  rows.forEach(r => Object.keys(r.metrics || {}).forEach(k => {
    const v = r.metrics[k];
    if (k === "CI" || k === "Threshold" || (v !== null && typeof v === "object")) return;
    if (!keys.includes(k)) keys.push(k);
  }));
  return keys;
}
function renderMetricsTable(t, rows) {
  t.innerHTML = "";
  const ok = rows.filter(r => r.metrics && !r.error);
  const keys = metricKeysOf(ok);
  if (!ok.length && !rows.length) { t.innerHTML = `<tr><td class="muted">No metrics available.</td></tr>`; return; }
  const thrCol = ok.some(r => r.threshold != null || (r.metrics && r.metrics.Threshold != null));
  const recalCol = ok.some(r => r.recalibrated != null);
  t.appendChild(el("tr", {}, `<th>Model</th>` +
    (thrCol ? `<th>${term("Threshold", "Decision threshold")}</th>` : "") +
    (recalCol ? `<th>${term("Recalibrated", "Recalibration")}</th>` : "") +
    keys.map(k => `<th class="nw">${term(METRIC_LABELS[k] || k, k)}</th>`).join("")));
  const span = keys.length + (thrCol ? 1 : 0) + (recalCol ? 1 : 0);
  rows.forEach(r => {
    if (r.error || !r.metrics) {
      t.appendChild(el("tr", {}, `<td><b>${esc(r.name)}</b></td><td colspan="${Math.max(span, 1)}" class="err-text">${esc(r.error || "No metrics.")}</td>`));
      return;
    }
    const m = r.metrics, ci = m.CI || {};
    const thr = r.threshold != null ? r.threshold : m.Threshold;
    const thrCell = thrCol ? `<td class="nw">${esc(fmt(thr, isNum(thr) ? 3 : null))}${r.threshold_method
      ? ` <span class="ci">(${esc(thrMethodLabel(r.threshold_method, r.name))})</span>` : ""}</td>` : "";
    const recalCell = recalCol ? `<td>${r.recalibrated ? "yes" : "no"}</td>` : "";
    t.appendChild(el("tr", {}, `<td><b>${esc(r.name)}</b></td>${thrCell}${recalCell}` + keys.map(k => {
      const v = m[k];
      if (k === "N") return `<td class="nw">${esc(fmt(v))}</td>`;
      return `<td class="nw">${ciHtml(v, ci[k], isNum(v) ? 3 : null)}</td>`;
    }).join("")));
  });
}

// adjusted odds ratios from the logistic model's training data
// d = {rows:[{term, or, ci_low, ci_high, p, variable, level}], n, events,
// n_parameters, epp, reference_levels, imputed, warning}
let OR_SEQ = 0;
function oddsRatioBlock(title, d, fname) {
  const box = el("div", { className: "or-block", style: "margin-top:16px" });
  const tid = `orTable${++OR_SEQ}`;
  const refs = d.reference_levels || {};
  const refTxt = Object.keys(refs).map(k => `${k} = ${refs[k]}`).join(", ");
  const imp = d.imputed || {};
  const impTxt = Object.keys(imp).map(k => `${k} (${fmt(imp[k], isNum(imp[k]) ? 3 : null)})`).join(", ");
  const rows = (d.rows || []).map(r => {
    const ref = r.variable != null ? refs[r.variable] : undefined;
    const label = r.level != null && r.variable != null
      ? `${esc(r.variable)}: ${esc(r.level)}${ref != null ? ` <span class="ci">vs ${esc(ref)}</span>` : ""}`
      : `${esc(r.term)} <span class="ci">per 1 unit</span>`;
    return `<tr><td>${label}</td><td class="nw">${ciHtml(r.or, [r.ci_low, r.ci_high], 3)}</td><td class="nw">${fmtP(r.p)}</td></tr>`;
  }).join("");
  box.innerHTML = `
    <div class="tbl-tools"><h4>${term("Adjusted odds ratios", "Odds ratios")}: ${esc(title)}</h4><span class="or-btns"></span></div>
    <div class="kv"><span>N <b>${fmtInt(d.n)}</b></span><span>Events <b>${fmtInt(d.events)}</b></span>
      <span>Parameters <b>${fmt(d.n_parameters)}</b></span>
      <span>${term("Events per parameter", "EPP")} <b>${fmt(d.epp)}</b></span></div>
    ${refTxt ? `<div class="small muted">Reference levels: ${esc(refTxt)}</div>` : ""}
    ${impTxt ? `<div class="small muted">Missing values filled with the training median or most common value: ${esc(impTxt)}</div>` : ""}
    ${d.warning ? `<div class="notices">${notice("warn", esc(d.warning))}</div>` : ""}
    ${rows ? `<div class="scrollx" style="margin-top:8px"><table id="${tid}" class="compact"><tr><th>Term</th><th>OR (95% CI)</th><th>p</th></tr>${rows}</table></div>`
      : `<div class="small muted" style="margin-top:6px">No odds ratios could be estimated.</div>`}`;
  if (rows) box.querySelector(".or-btns").appendChild(dlButton("CSV", () => dlTable(tid, `${fname}_odds_ratios.csv`)));
  return box;
}

// =====================================================================
// 4. test
// =====================================================================
$("testSource").addEventListener("change", e =>
  $("testUploadBox").style.display = e.target.value === "upload" ? "block" : "none");
$("pklUpload").addEventListener("change", e => {
  if (!e.target.files[0]) return;
  const fd = new FormData(); fd.append("file", e.target.files[0]);
  jform("/api/model/upload", fd).then(d => {
    if (d.error) toast(d.error, "err"); else { toast("Model uploaded.", "ok"); loadModels(); }
  });
  e.target.value = "";
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
const MODEL_META = {};
function modelLabel(name) {
  const m = MODEL_META[name];
  return m && m.model ? `${m.model}${m.outcome ? ` (${m.outcome})` : ""}` : name;
}
async function loadModels() {
  const d = await jget("/api/models");
  if (d.error) { toast(d.error, "err"); return; }
  d.models.forEach(m => { MODEL_META[m.name] = m.meta || {}; });
  const list = $("modelPickList"), pm = $("predModel"), hm = $("hlModel");
  if (pm) pm.innerHTML = ""; if (hm) hm.innerHTML = "";
  if (!d.models.length) { list.innerHTML = `<span class="muted small">Train Models to Display for Model Testing.</span>`;
    updateDeleteSelectedBtn(); return; }
  list.innerHTML = "";
  d.models.forEach(m => {
    const meta = m.meta || {};
    const taskPill = meta.task ? pill(meta.task) : "";
    const auc = meta.val_metrics && meta.val_metrics.AUROC != null ? `AUROC ${meta.val_metrics.AUROC}` : "";
    const when = fmtCreated(meta.created);
    const metaLine = [auc, when].filter(Boolean).join(" · ");
    const tip = `${m.name}${meta.outcome ? `, outcome: ${meta.outcome}` : ""}${when ? `, trained ${when}` : ""}`;
    list.appendChild(el("label", { title: tip }, `<input type="checkbox" class="tmdl" value="${esc(m.name)}">
      <span class="mdl-info">
        <div class="mdl-top"><b>${esc(meta.model || m.name)}</b>${taskPill}</div>
        ${meta.outcome ? `<span class="muted small mdl-sub">Outcome: ${esc(meta.outcome)}</span>` : ""}
        ${metaLine ? `<span class="muted small mdl-sub">${esc(metaLine)}</span>` : ""}
      </span>`));
    if (pm) pm.appendChild(opt(m.name));
    if (hm && meta.task === "binary") hm.appendChild(opt(m.name));
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
  const results = await Promise.all(names.map(n => jdelete(`/api/models/${encodeURIComponent(n)}`)));
  const failed = results.filter(r => r.error).length;
  toast(failed ? `Deleted ${names.length - failed} of ${names.length} models.` : "Model(s) deleted.",
        failed ? "err" : "ok");
  loadModels();
}
$("testBtn").addEventListener("click", () => runTests(null));
async function runTests(unlockReason) {
  const names = [...document.querySelectorAll(".tmdl:checked")].map(c => c.value);
  if (!names.length) { toast("Pick at least one model.", "err"); return; }
  const source = $("testSource").value;
  // empty threshold = each model keeps the one it chose at training
  const thrRaw = $("testThr").value.trim();
  if (thrRaw !== "" && !(+thrRaw > 0 && +thrRaw < 1)) { toast("Threshold must be between 0 and 1, or empty to use each model's saved threshold.", "err"); return; }
  const fd = new FormData();
  names.forEach(n => fd.append("models", n));
  fd.append("source", source);
  if (thrRaw !== "") fd.append("threshold", thrRaw);
  if (unlockReason) fd.append("unlock_reason", unlockReason);
  if (source === "upload") {
    if (!$("testCsv").files[0]) { toast("Choose a CSV.", "err"); return; }
    fd.append("file", $("testCsv").files[0]);
  }
  $("testBtn").disabled = true;
  let d;
  try { d = await jform("/api/test", fd); }
  finally { $("testBtn").disabled = false; }
  if (d.locked) {
    const reason = await askUnlockReason(d.new_models);
    if (reason) runTests(reason);
    return;
  }
  if (d.error) { toast(d.error, "err"); return; }
  document.querySelector('[data-tab="test"]').classList.add("done");
  renderTest(d);
}
function renderTest(d) {
  $("testResults").style.display = "block";
  $("testResults").scrollIntoView({ behavior: "smooth", block: "start" });
  const lock = d.test_lock;
  $("testLockNote").style.display = lock ? "block" : "none";
  if (lock) $("testLockNote").textContent =
    `The internal test partition is now locked to: ${lock.models.map(modelLabel).join(", ")}. ` +
    `Re-testing these (for example at another threshold) is fine; testing a different model on it ` +
    `will ask for a reason, which is logged.`;
  const ov = $("testOverlay"); ov.innerHTML = "";
  if (d.roc_png) ov.appendChild(plotBox(d.roc_png, "ROC overlay", "roc_overlay.png"));
  if (d.pr_png) ov.appendChild(plotBox(d.pr_png, "PR overlay", "pr_overlay.png"));

  renderMetricsTable($("testMetricsTable"), d.models.map(m => ({ name: m.name, metrics: m.metrics,
    error: m.error, threshold: m.threshold, threshold_method: m.threshold_method, recalibrated: m.recalibrated })));

  const dc = $("testDca"); dc.innerHTML = "";
  if (d.dca_png || d.dca) dc.appendChild(dcaCard(d));

  const pp = $("testPlots"); pp.innerHTML = "";
  d.models.forEach(m => {
    const meta = MODEL_META[m.name] || {};
    const canOR = meta.model === "LogisticRegression" && meta.task === "binary";
    const hasPlots = m.confusion_png || m.calibration_png || m.residuals_png ||
      m.pred_vs_actual_png || (m.correlations && m.correlations.length) || canOR;
    if (!hasPlots) return;
    const group = el("div", { className: "test-model-group" });
    group.appendChild(el("h4", { textContent: m.name }));
    const row = el("div", { className: "test-model-row" });
    if (m.confusion_png) row.appendChild(plotBox(m.confusion_png, "confusion", `${m.name}_confusion.png`));
    if (m.calibration_png) {
      const cb = plotBox(m.calibration_png, "calibration", `${m.name}_calibration.png`);
      if (m.coords && m.calibration_stats) cb.querySelector(".flex").appendChild(dlButton("PDF", () =>
        dlPdf("calibration_flexible", { ...m.coords, calibration_stats: m.calibration_stats, name: m.name }, `${m.name}_calibration`)));
      row.appendChild(cb);
    }
    if (m.calibration_stats) row.appendChild(calibrationSummary(m.calibration_stats));
    if (m.residuals_png) row.appendChild(plotBox(m.residuals_png, "residuals", "residuals.png"));
    if (m.pred_vs_actual_png) row.appendChild(plotBox(m.pred_vs_actual_png, "pred vs actual", "pva.png"));
    if (m.correlations && m.correlations.length) row.appendChild(correlationBox(m.name, m.correlations));
    group.appendChild(row);
    if (canOR) {
      const orBox = el("div");
      const b = dlButton("Show adjusted odds ratios", async () => {
        b.disabled = true;
        const r = await jget(`/api/model/${encodeURIComponent(m.name)}/odds-ratios`);
        b.disabled = false;
        if (r.error) { toast(r.error, "err"); return; }
        b.remove(); orBox.appendChild(oddsRatioBlock(m.name, r.odds_ratios || r, m.name));
      });
      b.style.marginTop = "10px";
      orBox.appendChild(b); group.appendChild(orBox);
    }
    pp.appendChild(group);
  });
}

// calibration summary with a plain reading of each number
// s = {slope, slope_ci, citl, citl_ci, oe_ratio, oe_ci, ici, e50, e90, reason}
function ciSide(v, ci, ideal, tol) {
  // -1 below ideal, 1 above, 0 consistent with ideal (ci covers it, or close without a ci)
  if (!isNum(v)) return null;
  if (hasCI(ci)) return ci[1] < ideal ? -1 : ci[0] > ideal ? 1 : 0;
  return v < ideal - tol ? -1 : v > ideal + tol ? 1 : 0;
}
const CAL_READ = {
  slope: { "-1": "Below 1: predictions are too extreme (high risks too high, low risks too low), a sign of overfitting.",
           "1": "Above 1: predictions are too modest, bunched toward the average.",
           "0": "Consistent with 1 (ideal): the spread of predicted risks looks right." },
  citl: { "-1": "Below 0: risks are overestimated on average.",
          "1": "Above 0: risks are underestimated on average.",
          "0": "Consistent with 0 (ideal): the average risk is about right." },
  oe: { "-1": "Below 1: fewer events happened than predicted.",
        "1": "Above 1: more events happened than predicted.",
        "0": "Consistent with 1 (ideal): predicted and observed event counts agree." },
};
function calibrationSummary(s) {
  const box = el("div", { className: "calsum" });
  const read = (key, side) => side == null ? "Could not be estimated." : CAL_READ[key][String(side)];
  const icipp = isNum(s.ici) ? `On average, predicted risk is ${(s.ici * 100).toFixed(1)} percentage points from the observed risk.` : "Could not be estimated.";
  const e = [isNum(s.e50) ? `E50 ${fmt(s.e50, 3)}` : "", isNum(s.e90) ? `E90 ${fmt(s.e90, 3)}` : ""].filter(Boolean).join(", ");
  box.innerHTML = `<div class="small muted" style="margin-bottom:6px"><b style="color:var(--ink)">Calibration summary</b> (brackets: 95% CI)</div>
    <table>
      <tr><td class="nw">${term("Calibration slope", "CalibrationSlope")}</td><td class="nw">${ciHtml(s.slope, s.slope_ci, 2)}</td>
        <td class="reading">${esc(read("slope", ciSide(s.slope, s.slope_ci, 1, 0.1)))}</td></tr>
      <tr><td class="nw">${term("CITL", "CalibrationInTheLarge")}</td><td class="nw">${ciHtml(s.citl, s.citl_ci, 2)}</td>
        <td class="reading">${esc(read("citl", ciSide(s.citl, s.citl_ci, 0, 0.1)))}</td></tr>
      <tr><td class="nw">${term("O:E ratio", "OE_ratio")}</td><td class="nw">${ciHtml(s.oe_ratio, s.oe_ci, 2)}</td>
        <td class="reading">${esc(read("oe", ciSide(s.oe_ratio, s.oe_ci, 1, 0.1)))}</td></tr>
      <tr><td class="nw">${term("ICI")}</td><td class="nw">${esc(fmt(s.ici, isNum(s.ici) ? 3 : null))}</td>
        <td class="reading">${esc(icipp)}${e ? ` <span class="nw">(${term("E50")} and ${term("E90")}: ${esc(e)})</span>` : ""}</td></tr>
    </table>
    ${s.reason ? `<div class="small muted" style="margin-top:6px">Not estimable: ${esc(s.reason)}.</div>` : ""}`;
  return box;
}

// decision curve card, one per test run (all binary models overlaid)
// d.dca = {thresholds, treat_all, models:[{name, net_benefit}]}
function dcaCard(d) {
  const box = el("div", { className: "test-model-group" });
  const dca = d.dca || {};
  box.innerHTML = `<h4>${term("Decision curve", "Decision curve")} (clinical usefulness)</h4>
    <p class="small" style="margin:4px 0 10px;max-width:820px;line-height:1.6">The curve shows
      ${term("net benefit", "Net benefit")}: the true positives a model finds minus its false positives, with each false
      alarm weighted by how much it costs at that ${term("threshold probability", "Threshold probability")} (the risk at which
      you would act, such as operating or monitoring more closely). A model is clinically useful at thresholds where its curve
      sits above both <b>treat all</b> and <b>treat none</b> (the line at zero). Read it only over the range of thresholds that
      makes sense for your decision; a model can have a high AUROC and still add nothing at the thresholds that matter.</p>
    ${d.dca_png ? `<img class="plot" src="${d.dca_png}" style="max-width:600px">` : ""}
    <div class="flex" style="margin-top:6px"></div>`;
  const btns = box.querySelector(".flex");
  if (d.dca_png) btns.appendChild(dlButton("PNG", () => dlPng(d.dca_png, "decision_curve.png")));
  if (d.dca) {
    btns.appendChild(dlButton("PDF", () => dlPdf("dca", dca, "decision_curve")));
    btns.appendChild(dlButton("CSV", () => dlBlob(dcaCsv(dca), "decision_curve.csv")));
  }
  return box;
}
function dcaCsv(dca) {
  const th = dca.thresholds || dca.threshold || [];
  const models = dca.models || [];
  const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const head = ["threshold", "treat_all", "treat_none", ...models.map(m => `net_benefit_${m.name}`)];
  const lines = th.map((t, i) => [t, (dca.treat_all || [])[i], 0,
    ...models.map(m => (m.net_benefit || [])[i])].map(cell).join(","));
  return head.map(cell).join(",") + "\n" + lines.join("\n");
}
function plotBox(src, label, fname) {
  const d = el("div", { className: "plotwrap" });
  d.innerHTML = `<div class="small muted">${esc(label)}</div><img class="plot" src="${src}" style="max-width:430px">
    <div class="flex"></div>`;
  d.querySelector(".flex").appendChild(dlButton("PNG", () => dlPng(src, fname)));
  return d;
}
function correlationBox(name, rows) {
  const d = el("div", { className: "plotwrap" });
  const body = rows.map(r => `<tr><td>${esc(r.feature)}</td><td>${num(r.pearson_r)}</td></tr>`).join("");
  d.innerHTML = `<details>
      <summary>${esc(name)}: view ${term("Pearson r")} table</summary>
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
  if (d.error) { toast(d.error, "err"); return; }
  if (!d.features) { w.innerHTML = `<span class="muted small">Uploaded model has no stored feature list. Use the JSON upload below.</span>`; return; }
  const grid = el("div", { className: "checklist" });
  // missing indicators are built from the blanks, not typed in
  const derived = new Set(d.derived_features || []);
  d.features.filter(f => !derived.has(f)).forEach(f => {
    const box = el("div"), inp = el("input", { className: "pf", type: "text" });
    inp.dataset.f = f;
    box.append(el("label", { textContent: f }), inp);
    grid.appendChild(box);
  });
  w.appendChild(grid);
  const go = el("button", { textContent: "Predict outcome" });
  go.onclick = () => predictForm(name);
  w.appendChild(el("div", { style: "margin-top:10px" })).appendChild(go);
}
async function predictForm(name) {
  const row = {};
  document.querySelectorAll(".pf").forEach(i => {
    const v = i.value.trim();
    if (v !== "") row[i.dataset.f] = isNaN(+v) ? v : +v;      // blank stays missing
  });
  const d = await jpost("/api/predict-single", { model: name, row });
  showPred(d);
}
async function predictUpload() {
  if (!$("predPkl").files[0] || !$("predJson").files[0]) { toast("Choose model and JSON.", "err"); return; }
  const fd = new FormData(); fd.append("model_file", $("predPkl").files[0]); fd.append("json_file", $("predJson").files[0]);
  showPred(await jform("/api/predict-single", fd));
}
function showPred(d) {
  if (d.error) { toast(d.error, "err"); return; }
  let h = `<div class="card"><div class="lbl">Prediction</div><div class="big">${esc(d.prediction)}</div>`;
  if (d.probability != null) h += `<div class="small">probability ${(d.probability*100).toFixed(1)}%</div>`;
  h += `<div class="small muted" style="margin-top:6px">${esc(d.explanation)}</div></div>`;
  $("predResult").innerHTML = h;
}

// =====================================================================
// 5. spline
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
// 6. epi
// =====================================================================
let EPI_COLS = [], EPI_LEVELS = {};
async function initEpi() {
  initSampleSize();
  const d = await jget("/api/train/columns");
  if (d.error) { $("epiEmpty").style.display = "block"; $("epiBody").style.display = "none"; $("epiEmpty").textContent = d.error; return; }
  $("epiEmpty").style.display = "none"; $("epiBody").style.display = "block";
  clearStale("epi", ["kmResult", "coxResult", "cifResult", "hlResult", "ssFillInfo"]);
  EPI_COLS = d.columns;
  const allNum = d.columns.map(c => c.name);
  keepSel($("kmTime"), () => fillSel($("kmTime"), allNum));
  keepSel($("kmEvent"), () => fillSel($("kmEvent"), allNum));
  keepSel($("kmGroup"), () => fillSel($("kmGroup"), allNum, true));
  keepSel($("cifTime"), () => fillSel($("cifTime"), allNum));
  keepSel($("cifEvent"), () => fillSel($("cifEvent"), allNum));
  keepSel($("cifGroup"), () => fillSel($("cifGroup"), allNum, true));
  // sensible first picks for the sample data style of column names
  guessSel($("kmTime"), /time|days|duration|follow/i); guessSel($("kmEvent"), /event|death|status/i);
  guessSel($("cifTime"), /time|days|duration|follow/i); guessSel($("cifEvent"), /event|status|death/i);
  const prevCov = multiValues($("coxCovList"));
  fillChecklist($("coxCovList"), allNum);
  $("coxCovList").querySelectorAll("input").forEach(cb => { cb.checked = prevCov.includes(cb.value); });
  // category levels for the cox reference pickers
  const s = await jget("/api/summary");
  EPI_LEVELS = {};
  if (!s.error) (s.categorical || []).forEach(v => { EPI_LEVELS[v.variable] = v.categories.map(c => String(c.category)); });
  renderCoxCats();
  loadModels();
  ssLoadData(d);
}
function keepSel(sel, fill) { const v = sel.value; fill(); if (v && [...sel.options].some(o => o.value === v)) sel.value = v; }
function guessSel(sel, re) {
  if (sel.dataset.touched) return;
  const o = [...sel.options].find(x => x.value && re.test(x.value));
  if (o) sel.value = o.value;
}
["kmTime", "kmEvent", "cifTime", "cifEvent"].forEach(id => $(id).addEventListener("change", e => { e.target.dataset.touched = "1"; }));
function fillSel(sel, names, withNone) {
  sel.innerHTML = withNone ? `<option value="">None</option>` : "";
  names.forEach(n => sel.appendChild(opt(n)));
}
$("epiJump").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-jump]"); if (!a) return;
  const target = $(a.dataset.jump);
  if (target.offsetParent === null) { toast("Confirm a dataset to use this tool.", ""); $("epiEmpty").scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  target.scrollIntoView({ behavior: "smooth", block: "start" });
});

async function runEpi2x2() {
  const body = { a: +$("ea").value, b: +$("eb").value, c: +$("ec").value, d: +$("ed").value };
  const r = await jpost("/api/epi/or", body);
  const n = await jpost("/api/epi/nnt", body);
  if (r.error || n.error) { $("epi2x2Result").innerHTML = ""; toast(r.error || n.error, "err"); return; }
  const corr = r.correction_applied || r.corrected;
  const corrTag = corr ? ` <span class="tag warn" title="0.5 added to every cell for this estimate only">corrected</span>` : "";
  const nntCi = (r.nnt_ci && r.nnt_ci.text) || (n.nnt_ci && n.nnt_ci.text) || "";
  const nntVal = n.value ?? r.nnt;
  const notes = [...(r.notes || [])];
  if (corr && !notes.some(x => /Haldane|0\.5/.test(x)))
    notes.push("A cell is zero, so 0.5 was added to every cell for the odds ratio and risk ratio only (Haldane-Anscombe). Risks, risk difference, NNT and p-values use the observed counts.");
  $("epi2x2Result").innerHTML = `
    <div class="cards">
      <div class="card"><div class="lbl">${term("Odds Ratio")}${corrTag}</div><div class="big">${esc(fmt(r.odds_ratio))}</div><div class="small">95% CI ${esc(fmt(r.or_ci && r.or_ci[0]))} to ${esc(fmt(r.or_ci && r.or_ci[1]))}</div></div>
      <div class="card"><div class="lbl">${term("Risk Ratio")}${corrTag}</div><div class="big">${esc(fmt(r.risk_ratio))}</div><div class="small">95% CI ${esc(fmt(r.rr_ci && r.rr_ci[0]))} to ${esc(fmt(r.rr_ci && r.rr_ci[1]))}</div></div>
      <div class="card"><div class="lbl">${term("Risk difference")}</div><div class="big">${esc(fmt(r.abs_risk_diff))}</div><div class="small">${hasCI(r.rd_ci) ? `95% CI ${esc(fmt(r.rd_ci[0]))} to ${esc(fmt(r.rd_ci[1]))}` : ""}</div>
        <div class="small muted">risk ${fmtPct(r.risk_exposed)}% exposed vs ${fmtPct(r.risk_unexposed)}% unexposed</div></div>
      <div class="card"><div class="lbl">${term(n.measure || "NNT/NNH", n.measure || "NNT")}</div><div class="big">${esc(fmt(nntVal))}</div><div class="small">${esc(nntCi)}</div></div>
      <div class="card"><div class="lbl">${term("Chi-square")} p-value</div><div class="big">${esc(fmtP(r.p_value))}</div><div class="small">χ² = ${esc(fmt(r.chi2))}</div></div>
      <div class="card"><div class="lbl">${term("Fisher exact", "Fisher exact")} p-value</div><div class="big">${esc(fmtP(r.fisher_p_value))}</div><div class="small">two-sided</div></div>
    </div>
    <div class="small muted" style="margin-top:8px">${esc(n.interpretation || "")}</div>
    ${r.chi2_method ? `<div class="small muted" style="margin-top:4px">Chi-square: ${esc(r.chi2_method)}.</div>` : ""}
    ${notes.length ? `<div class="notices">${notes.map(x => notice("warn", esc(x))).join("")}</div>` : ""}`;
}

async function runKM() {
  const tp = parseNumberList($("kmTimes").value);
  if (tp === undefined) { toast("Time points must be numbers separated by commas, for example 30, 90, 365.", "err"); return; }
  const d = await jpost("/api/epi/km", { time: $("kmTime").value, event: $("kmEvent").value,
    group: $("kmGroup").value || null, time_points: tp });
  if (d.error) { toast(d.error, "err"); return; }
  const med = (c) => {
    if (c.median_survival == null) return "not reached";
    const ci = c.median_ci || [];
    return `${fmt(c.median_survival)} (${fmt(ci[0])} to ${ci[1] == null ? "not reached" : fmt(ci[1])})`;
  };
  let h = `<img class="plot" src="${d.png}" style="max-width:640px">
    <div class="flex" style="margin-top:6px" id="kmPlotBtns"></div>
    <div class="tbl-tools"><h4>By group</h4><span id="kmGroupBtns"></span></div>
    <div class="scrollx"><table id="kmGroupTable" class="compact"><tr><th>Group</th><th>N</th><th>Events</th><th class="nw">${term("Median survival")} (95% CI)</th></tr>`;
  d.curves.forEach(c => h += `<tr><td>${esc(c.name)}</td><td>${fmtInt(c.n)}</td><td>${c.events == null ? "n/a" : fmtInt(c.events)}</td><td class="nw">${esc(med(c))}</td></tr>`);
  h += `</table></div>`;
  if (d.logrank) {
    const lr = d.logrank;
    h += `<div class="small" style="margin-top:8px">${term("Log-rank")} test${lr.n_groups ? ` across ${lr.n_groups} groups` : ""}:
      χ² = ${esc(fmt(lr.test_statistic))}${lr.df != null ? ` on ${esc(lr.df)} df` : ""}, p = ${esc(fmtP(lr.p_value))}</div>`;
  } else if (d.curves.length === 1 && $("kmGroup").value) {
    h += `<div class="small muted" style="margin-top:8px">Only one group has data, so there is no log-rank test.</div>`;
  }
  const at = [];
  d.curves.forEach(c => (c.survival_at || []).forEach(p => at.push([c.name, p])));
  if (at.length) {
    h += `<div class="tbl-tools"><h4>Survival at time points</h4><span id="kmAtBtns"></span></div>
      <div class="scrollx"><table id="kmAtTable" class="compact"><tr><th>Group</th><th>Time</th><th>${term("At risk")}</th><th class="nw">Survival, % (95% CI)</th></tr>` +
      at.map(([g, p]) => `<tr><td>${esc(g)}</td><td>${esc(fmt(p.time))}</td><td>${p.at_risk == null ? "n/a" : fmtInt(p.at_risk)}</td>
        <td class="nw">${esc(pctCI(p.survival, p.lower, p.upper))}</td></tr>`).join("") + `</table></div>`;
  }
  if (d.ci_method) h += `<div class="hint">${esc(d.ci_method)}.</div>`;
  $("kmResult").innerHTML = h;
  $("kmPlotBtns").appendChild(dlButton("PNG", () => dlPng(d.png, "kaplan_meier.png")));
  $("kmPlotBtns").appendChild(dlButton("PDF", () => dlPdf("km", { curves: d.curves, risk_table: d.risk_table }, "kaplan_meier")));
  $("kmGroupBtns").appendChild(dlButton("CSV", () => dlTable("kmGroupTable", "km_groups.csv")));
  if (at.length) $("kmAtBtns").appendChild(dlButton("CSV", () => dlTable("kmAtTable", "km_survival_at_times.csv")));
}

// cox: ticked covariates, each optionally categorical with a reference level
function renderCoxCats() {
  const box = $("coxCatBox");
  const cov = multiValues($("coxCovList"));
  const prev = {};
  box.querySelectorAll(".coxcat").forEach(r => {
    prev[r.dataset.col] = { cat: r.querySelector("input[type=checkbox]").checked, ref: (r.querySelector("select,input[type=text]") || {}).value };
  });
  if (!cov.length) { box.innerHTML = `<div class="small muted" style="padding:8px 0">Tick covariates first.</div>`; return; }
  const types = Object.fromEntries(EPI_COLS.map(c => [c.name, c.type]));
  box.innerHTML = "";
  const tbl = el("table", { className: "compact" });
  tbl.innerHTML = `<tr><th>Covariate</th><th>Categorical</th><th>Reference level</th></tr>`;
  cov.forEach(c => {
    const p = prev[c] || { cat: types[c] === "categorical", ref: "" };
    const levels = EPI_LEVELS[c];
    const refCtl = levels && levels.length
      ? `<select${p.cat ? "" : " disabled"}><option value="">First level (default)</option>${levels.map(l => `<option value="${esc(l)}"${l === p.ref ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>`
      : `<input type="text" placeholder="first level"${p.cat ? "" : " disabled"} value="${esc(p.ref || "")}">`;
    const tr = el("tr", { className: "coxcat" });
    tr.dataset.col = c;
    tr.innerHTML = `<td>${esc(c)} <span class="muted small">${esc(types[c] || "")}</span></td>
      <td><input type="checkbox"${p.cat ? " checked" : ""} aria-label="Treat ${esc(c)} as categorical"></td><td>${refCtl}</td>`;
    tr.querySelector("input[type=checkbox]").addEventListener("change", (e) => {
      tr.querySelector("select,input[type=text]").disabled = !e.target.checked;
    });
    tbl.appendChild(tr);
  });
  box.appendChild(tbl);
}
$("coxCovList").addEventListener("change", renderCoxCats);
async function runCox() {
  const cov = multiValues($("coxCovList"));
  if (!cov.length) { toast("Pick covariates.", "err"); return; }
  const categorical = [], reference = {};
  $("coxCatBox").querySelectorAll(".coxcat").forEach(r => {
    if (!r.querySelector("input[type=checkbox]").checked) return;
    categorical.push(r.dataset.col);
    const ref = (r.querySelector("select,input[type=text]").value || "").trim();
    if (ref) reference[r.dataset.col] = ref;
  });
  const d = await jpost("/api/epi/cox", { time: $("kmTime").value, event: $("kmEvent").value,
    covariates: cov, categorical, reference });
  if (d.error) { toast(d.error, "err"); return; }
  const refs = d.reference_levels || {};
  const refTxt = Object.keys(refs).map(k => `${k} = ${refs[k]}`).join(", ");
  let h = `<div class="kv"><span>N <b>${fmtInt(d.n)}</b></span><span>Events <b>${fmtInt(d.n_events)}</b></span>
      ${d.n_excluded ? `<span>Left out for missing values <b>${fmtInt(d.n_excluded)}</b></span>` : ""}
      <span>${term("Concordance", "concordance")} <b>${esc(fmt(d.concordance))}</b></span></div>
    ${refTxt ? `<div class="small muted">Reference levels: ${esc(refTxt)}</div>` : ""}
    <div class="tbl-tools"><h4>Hazard ratios</h4><span id="coxBtns"></span></div>
    <div class="scrollx"><table id="coxTable" class="compact"><tr><th>Covariate</th><th class="nw">${term("HR")} (95% CI)</th><th>p</th><th class="nw">${term("PH test", "Proportional hazards test")} p</th></tr>`;
  d.rows.forEach(r => h += `<tr><td>${esc(r.covariate)}</td><td class="nw">${ciHtml(r.hazard_ratio, [r.hr_lower, r.hr_upper], 3)}</td>
    <td class="nw">${fmtP(r.p_value)}</td><td class="nw">${fmtP(r.ph_p_value)}</td></tr>`);
  h += `</table></div>`;
  const ph = d.ph_test;
  if (ph && (ph.terms || ph.global)) {
    h += `<div class="tbl-tools"><h4>${term("Proportional hazards test", "Proportional hazards test")}</h4><span id="coxPhBtns"></span></div>
      <div class="scrollx"><table id="coxPhTable" class="compact"><tr><th>Term</th><th>χ²</th><th>df</th><th>p</th></tr>` +
      (ph.terms || []).map(t => `<tr><td>${esc(t.variable)}</td><td>${esc(fmt(t.test_statistic))}</td><td>${esc(fmt(t.df))}</td><td class="nw">${fmtP(t.p_value)}</td></tr>`).join("") +
      (ph.global ? `<tr><td><b>Global</b></td><td><b>${esc(fmt(ph.global.test_statistic))}</b></td><td><b>${esc(fmt(ph.global.df))}</b></td><td class="nw"><b>${fmtP(ph.global.p_value)}</b></td></tr>` : "") +
      `</table></div>${ph.method ? `<div class="hint">${esc(ph.method)}.</div>` : ""}`;
  }
  if (d.ph_notes && d.ph_notes.length) h += `<div class="notices">${d.ph_notes.map(x => notice("warn", esc(x))).join("")}</div>`;
  else if (ph && ph.global && isNum(ph.global.p_value)) h += `<div class="notices">${notice("ok", "No covariate shows clear evidence against proportional hazards (all p of 0.05 or more).")}</div>`;
  $("coxResult").innerHTML = h;
  $("coxBtns").appendChild(dlButton("CSV", () => dlTable("coxTable", "cox_hazard_ratios.csv")));
  if ($("coxPhBtns")) $("coxPhBtns").appendChild(dlButton("CSV", () => dlTable("coxPhTable", "cox_ph_test.csv")));
}

// competing risks: aalen-johansen cumulative incidence next to naive 1 minus km
async function runCIF() {
  const tp = parseNumberList($("cifTimes").value);
  if (tp === undefined) { toast("Time points must be numbers separated by commas, for example 1, 2, 5.", "err"); return; }
  const code = $("cifCode").value.trim();
  if (code === "" || !isFinite(+code)) { toast("Enter the code of the event of interest (usually 1).", "err"); return; }
  const d = await jpost("/api/epi/cif", { time: $("cifTime").value, event: $("cifEvent").value,
    event_of_interest: +code, group: $("cifGroup").value || null, time_points: tp });
  if (d.error) { toast(d.error, "err"); return; }
  const curves = d.curves || [];
  let h = (d.notes && d.notes.length) ? `<div class="notices" style="margin-top:0">${d.notes.map(x => notice("info", esc(x))).join("")}</div>` : "";
  h += `${d.png ? `<img class="plot" src="${d.png}" style="max-width:640px;margin-top:10px">
    <div class="flex" style="margin-top:6px" id="cifPlotBtns"></div>` : ""}`;
  if (curves.some(c => c.n != null)) {
    h += `<div class="tbl-tools"><h4>Events by group</h4></div><div class="scrollx"><table class="compact"><tr><th>Group</th><th>N</th><th>Events of interest</th><th>Competing events</th><th>Censored</th></tr>` +
      curves.map(c => `<tr><td>${esc(c.name)}</td><td>${fmtInt(c.n)}</td><td>${fmtInt(c.events)}</td><td>${fmtInt(c.competing_events)}</td><td>${fmtInt(c.censored)}</td></tr>`).join("") + `</table></div>`;
  }
  const g = d.gray_test;
  h += `<div class="small" style="margin-top:8px">${g
    ? `${term("Gray's test")}: χ² = ${esc(fmt(g.test_statistic))} on ${esc(fmt(g.df))} df, p = ${esc(fmtP(g.p_value))}`
    : `${term("Gray's test")} needs a group column with two or more groups.`}</div>`;
  const rows = [];
  curves.forEach(c => (c.at_times || []).forEach(p => rows.push([c.name, p])));
  if (rows.length) {
    h += `<div class="tbl-tools"><h4>Cumulative incidence at time points</h4><span id="cifBtns"></span></div>
      <div class="scrollx"><table id="cifTable" class="compact"><tr><th>Group</th><th>Time</th><th>${term("At risk")}</th>
        <th class="nw">${term("Cumulative incidence")}, % (95% CI)</th><th class="nw">${term("Naive 1 minus KM")}, %</th></tr>` +
      rows.map(([gname, p]) => `<tr><td>${esc(gname)}</td><td>${esc(fmt(p.time))}</td><td>${p.at_risk == null ? "n/a" : fmtInt(p.at_risk)}</td>
        <td class="nw">${esc(pctCI(p.cif, p.lower, p.upper))}</td><td class="nw">${esc(fmtPct(p.naive))}${isNum(p.naive) && isNum(p.cif) && p.naive - p.cif >= 0.005
          ? ` <span class="ci">(+${((p.naive - p.cif) * 100).toFixed(1)})</span>` : ""}</td></tr>`).join("") + `</table></div>
      <div class="hint">The number in brackets after the naive estimate is how many percentage points too high it is.</div>`;
  }
  $("cifResult").innerHTML = h;
  if (d.png) {
    $("cifPlotBtns").appendChild(dlButton("PNG", () => dlPng(d.png, "cumulative_incidence.png")));
    $("cifPlotBtns").appendChild(dlButton("PDF", () => dlPdf("cif", { curves: d.curves, naive: d.naive, risk_table: d.risk_table }, "cumulative_incidence")));
  }
  if (rows.length) $("cifBtns").appendChild(dlButton("CSV", () => dlTable("cifTable", "cumulative_incidence.csv")));
}
async function runHL(unlockReason) {
  if (!$("hlModel").value) { toast("Pick a binary model.", "err"); return; }
  const body = { model: $("hlModel").value };
  if (typeof unlockReason === "string" && unlockReason) body.unlock_reason = unlockReason;
  const d = await jpost("/api/epi/hl", body);
  if (d.locked) {
    const reason = await askUnlockReason(d.new_models);
    if (reason) runHL(reason);
    return;
  }
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

// ---------- sample size (riley et al.), works without a dataset ----------
let SS_MODE = "binary", SS_INIT = false;
const SS_PERF = {
  cstatistic: { lbl: "Anticipated C statistic", ph: "e.g. 0.75", key: "C statistic",
    hint: "The AUROC you expect, from earlier models for this outcome. It is turned into a Cox-Snell R² for the calculation." },
  csrsquared: { lbl: "Anticipated Cox-Snell R²", ph: "e.g. 0.05", key: "Cox-Snell R2",
    hint: "From earlier models for this outcome. If unknown, Riley et al. suggest 15% of its maximum (shown in the notes after you calculate)." },
  nagrsquared: { lbl: "Anticipated Nagelkerke R²", ph: "e.g. 0.15", key: "Nagelkerke R2",
    hint: "From earlier models for this outcome; it is turned into a Cox-Snell R² for the calculation." },
};
function initSampleSize() {
  if (SS_INIT) return; SS_INIT = true;
  $("ssMode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]"); if (!b) return;
    SS_MODE = b.dataset.mode;
    $("ssMode").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    $("ssBinary").style.display = SS_MODE === "binary" ? "" : "none";
    $("ssContinuous").style.display = SS_MODE === "continuous" ? "" : "none";
    $("ssMarginBox").style.display = SS_MODE === "binary" ? "" : "none";
    ssSyncData();
    $("ssResult").innerHTML = "";
  });
  $("ssPerf").addEventListener("change", ssPerfChange); ssPerfChange();
  $("ssFillBtn").addEventListener("click", ssFill);
  $("ssBtn").addEventListener("click", runSampleSize);
}
function ssPerfChange() {
  const p = SS_PERF[$("ssPerf").value];
  $("ssPerfLbl").innerHTML = term(p.lbl, p.key);
  $("ssPerfVal").placeholder = p.ph;
  $("ssPerfHint").textContent = p.hint;
}
let SS_HAS_DATA = false;
function ssLoadData(d) {
  SS_HAS_DATA = !d.error;
  if (SS_HAS_DATA) {
    const bins = (d.outcomes || []).filter(o => (o.task || o.type) === "binary").map(o => o.name);
    keepSel($("ssOutcome"), () => fillSel($("ssOutcome"), bins));
    const trainOut = $("outcomeSel").value;
    if (!$("ssOutcome").dataset.touched && bins.includes(trainOut)) $("ssOutcome").value = trainOut;
    const prev = multiValues($("ssExclude"));
    const fromTrain = $("excludeSel") ? multiValues($("excludeSel")) : [];
    fillChecklist($("ssExclude"), d.columns.map(c => c.name));
    const pick = prev.length ? prev : fromTrain;
    $("ssExclude").querySelectorAll("input").forEach(cb => { cb.checked = pick.includes(cb.value); });
  }
  ssSyncData();
}
$("ssOutcome").addEventListener("change", e => { e.target.dataset.touched = "1"; });
function ssSyncData() {
  // comparing against the data is for binary outcomes only
  $("ssDataBox").style.display = SS_HAS_DATA && SS_MODE === "binary" && $("ssOutcome").options.length ? "block" : "none";
}
function ssExcludeList() { return multiValues($("ssExclude")).filter(c => c !== $("ssOutcome").value); }
async function ssFill() {
  const outcome = $("ssOutcome").value;
  if (!outcome) { toast("Pick the outcome first.", "err"); return; }
  const q = new URLSearchParams({ outcome });
  const ex = ssExcludeList();
  if (ex.length) q.set("exclude", ex.join(","));
  const d = await jget(`/api/samplesize/defaults?${q}`);
  if (d.error) { toast(d.error, "err"); return; }
  if (isNum(d.prevalence)) $("ssPrev").value = +d.prevalence.toFixed(4);
  if (isNum(d.n_parameters)) $("ssParams").value = d.n_parameters;
  $("ssFillInfo").innerHTML = `From your data: <b>${fmtInt(d.n)}</b> patients, <b>${fmtInt(d.events)}</b> events
    (outcome rate ${fmtPct(d.prevalence)}%), <b>${fmt(d.n_parameters)}</b> candidate parameters. Now enter the expected performance.`;
  toast("Filled from your data.", "ok");
  $("ssPerfVal").focus();
}
function ssNum(id, name, lo, hi, { int = false, loOpen = true, hiOpen = true } = {}) {
  const raw = $(id).value.trim(), v = Number(raw);
  const bad = raw === "" || !isFinite(v) || (int && !Number.isInteger(v)) ||
    (lo != null && (loOpen ? v <= lo : v < lo)) || (hi != null && (hiOpen ? v >= hi : v > hi));
  if (bad) { $(id).focus(); throw new Error(name); }
  return v;
}
async function runSampleSize() {
  let body;
  try {
    const adv = { shrinkage: ssNum("ssShrink", "Target shrinkage must be between 0 and 1.", 0, 1),
                  delta: ssNum("ssDelta", "Delta must be between 0 and 1.", 0, 1) };
    if (SS_MODE === "binary") {
      const perf = $("ssPerf").value;
      body = { outcome_type: "binary",
        prevalence: ssNum("ssPrev", "Outcome rate must be between 0 and 1 (for example 0.05).", 0, 1),
        n_parameters: ssNum("ssParams", "Candidate parameters must be a whole number of 1 or more.", 1, null, { int: true, loOpen: false }),
        ...adv, margin: ssNum("ssMargin", "Margin must be between 0 and 1.", 0, 1) };
      body[perf] = perf === "cstatistic"
        ? ssNum("ssPerfVal", "The C statistic must be above 0.5 and below 1.", 0.5, 1)
        : ssNum("ssPerfVal", "R² must be between 0 and 1.", 0, 1);
      if (SS_HAS_DATA && $("ssCompare").checked && $("ssOutcome").value)
        body.compare = { outcome: $("ssOutcome").value, exclude: ssExcludeList() };
    } else {
      body = { outcome_type: "continuous",
        rsquared: ssNum("ssR2", "R² must be between 0 and 1.", 0, 1),
        n_parameters: ssNum("ssParamsC", "Candidate parameters must be a whole number of 1 or more.", 1, null, { int: true, loOpen: false }),
        intercept: ssNum("ssIntercept", "Enter the average outcome value.", null, null),
        sd: ssNum("ssSd", "The standard deviation must be above 0.", 0, null), ...adv };
    }
  } catch (e) { toast(e.message, "err"); return; }
  $("ssBtn").disabled = true;
  let d;
  try { d = await jpost("/api/samplesize", body); }
  finally { $("ssBtn").disabled = false; }
  if (d.error) { $("ssResult").innerHTML = `<div class="notices">${notice("err", esc(d.error))}</div>`; toast(d.error, "err"); return; }
  renderSampleSize(d, body.outcome_type);
}
function renderSampleSize(d, kind) {
  const r = d.result || {}, cmp = d.compare;
  const bin = kind === "binary";
  const cards = [["Minimum patients", fmtInt(r.min_n)]];
  if (bin) cards.push(["Minimum events", fmtInt(r.min_events)], [term("Events per parameter", "EPP"), fmt(r.epp)]);
  else cards.push(["Patients per parameter", fmt(r.spp)]);
  cards.push([term("Expected shrinkage", "Shrinkage"), fmt(r.final_shrinkage)]);
  let h = `<div class="cards">${cards.map(([l, v]) => `<div class="card"><div class="lbl">${l}</div><div class="big">${esc(v)}</div></div>`).join("")}</div>`;
  if (cmp) {
    const ok = cmp.adequate;
    const short = [cmp.shortfall ? `${fmtInt(cmp.shortfall)} patients` : "", cmp.events_shortfall ? `${fmtInt(cmp.events_shortfall)} events` : ""].filter(Boolean).join(" and ");
    h += `<div class="verdict ${ok ? "ok" : "err"}"><div class="v-icon">${ok ? "&check;" : "!"}</div><div>
      <div class="v-title">${ok ? "Your data are big enough" : "Your data are not big enough"}</div>
      <div class="small">Your data have <b>${fmtInt(cmp.n)}</b> patients and <b>${fmtInt(cmp.events)}</b> events
        (${fmt(cmp.epp)} events per parameter). The minimum is <b>${fmtInt(r.min_n)}</b> patients${bin ? ` and <b>${fmtInt(r.min_events)}</b> events` : ""}.
        ${!ok && short ? `You are short by ${short}.` : ""}</div>
      ${(cmp.notes || []).length ? `<ul class="small" style="margin:6px 0 0;padding-left:18px">${cmp.notes.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    </div></div>`;
  }
  const crit = r.criteria || [];
  h += `<div class="tbl-tools"><h4>Criteria (the largest sets the minimum)</h4><span id="ssBtns"></span></div><div class="scrollx">
    <table id="ssTable"><tr><th>Criterion</th><th>Patients</th>${bin ? `<th>Events</th><th>${term("EPP")}</th><th>What it protects</th>` : `<th>Per parameter</th><th>${term("Shrinkage")}</th>`}</tr>` +
    crit.map(c => `<tr${c.n === r.min_n ? ' class="row-good"' : ""}><td>${esc(c.name)}${c.n === r.min_n ? ` <span class="tag ok">sets minimum</span>` : ""}</td><td class="nw">${fmtInt(c.n)}</td>` +
      (bin ? `<td class="nw">${fmtInt(c.events)}</td><td class="nw">${esc(fmt(c.epp))}</td><td class="small">${esc(c.description || "")}</td>`
           : `<td class="nw">${esc(fmt(c.spp))}</td><td class="nw">${esc(fmt(c.shrinkage))}</td>`) + `</tr>`).join("") +
    `</table></div>`;
  if ((r.notes || []).length) h += `<ul class="small muted" style="margin:10px 0 0;padding-left:18px;line-height:1.6">${r.notes.map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;
  if (r.reference) h += `<div class="small muted" style="margin-top:6px">Reference: ${esc(r.reference)}</div>`;
  $("ssResult").innerHTML = h;
  $("ssBtns").appendChild(dlButton("CSV", () => dlTable("ssTable", "sample_size.csv")));
}

// =====================================================================
// leakage demo (teaching)
// =====================================================================
let teachES = null, TEACH_RESULT = null;
async function initTeach() {
  const d = await jget("/api/train/columns");
  if (d.error || !(d.columns || []).length) {
    $("teachEmpty").style.display = "block"; $("teachBody").style.display = "none";
    $("teachEmpty").textContent = d.error && !/session/i.test(d.error) ? d.error : "Confirm a dataset first."; return;
  }
  $("teachEmpty").style.display = "none"; $("teachBody").style.display = "block";
  if (teachES) return;
  if (clearStale.seen.teach !== State.dataVersion) {
    clearStale("teach", ["teachError", "teachNotes", "teachCards", "teachTable", "teachPlot"]);
    $("teachResults").style.display = "none"; $("teachLog").style.display = "none"; $("teachStatus").textContent = "";
    $("teachBar").style.width = "0"; TEACH_RESULT = null;
  }              // a run is streaming, leave the form as it is
  const bins = (d.outcomes || []).filter(o => (o.task || o.type) === "binary").map(o => o.name);
  keepSel($("teachOutcome"), () => fillSel($("teachOutcome"), bins));
  if (!$("teachOutcome").dataset.touched && bins.includes($("outcomeSel").value)) $("teachOutcome").value = $("outcomeSel").value;
  const prev = multiValues($("teachExclude"));
  const fromTrain = multiValues($("excludeSel"));
  fillChecklist($("teachExclude"), d.columns.map(c => c.name));
  const pick = prev.length ? prev : fromTrain;
  $("teachExclude").querySelectorAll("input").forEach(cb => { cb.checked = pick.includes(cb.value); });
  if (!bins.length) $("teachError").innerHTML = `<div class="notices">${notice("warn", "This dataset has no binary outcome column, which the demo needs.")}</div>`;
}
$("teachOutcome").addEventListener("change", e => { e.target.dataset.touched = "1"; });
function teachLog(s) { const l = $("teachLog"); l.textContent += s + "\n"; l.scrollTop = l.scrollHeight; }
function setTeachActive(on) {
  $("teachBtn").style.display = on ? "none" : "";
  $("teachCancelBtn").style.display = on ? "" : "none";
  $("teachBarWrap").classList.toggle("running", on);
}
$("teachBtn").addEventListener("click", runTeach);
$("teachCancelBtn").addEventListener("click", () => {
  if (!teachES) return;
  teachES.close(); teachES = null; setTeachActive(false);
  $("teachStatus").textContent = "";
  teachLog("Cancelled.");
  toast("Demo cancelled.", "");
});
async function runTeach() {
  const outcome = $("teachOutcome").value;
  if (!outcome) { toast("Pick a binary outcome.", "err"); return; }
  const n_seeds = parseInt($("teachSeeds").value, 10), max_rows = parseInt($("teachMaxRows").value, 10);
  if (!(n_seeds >= 1 && n_seeds <= 5)) { toast("Repeats must be a whole number from 1 to 5.", "err"); return; }
  if (!(max_rows >= 200)) { toast("Max rows must be at least 200.", "err"); return; }
  const exclude = multiValues($("teachExclude")).filter(c => c !== outcome);
  $("teachError").innerHTML = "";
  $("teachLog").style.display = "block"; $("teachLog").textContent = "";
  $("teachResults").style.display = "none";
  $("teachBar").style.width = "0";
  setTeachActive(true);
  $("teachStatus").textContent = "Starting…";
  const cfg = await jpost("/api/teaching", { outcome, exclude, model: $("teachModel").value, n_seeds, max_rows });
  if (cfg.error) {
    setTeachActive(false); $("teachStatus").textContent = ""; $("teachLog").style.display = "none";
    $("teachError").innerHTML = `<div class="notices">${notice("err", `<b>The demo could not start.</b> ${esc(cfg.error)}`)}</div>`;
    toast("The demo could not start.", "err"); return;
  }
  const t0 = Date.now();
  teachLog(`Leakage demo: ${$("teachModel").selectedOptions[0].text}, outcome ${outcome}, ${plural(n_seeds, "repeat")}.`);
  const es = new EventSource("/api/teaching/stream");
  teachES = es;
  const tick = setInterval(() => { if (teachES === es) $("teachStatus").textContent = `${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
  const stop = () => { clearInterval(tick); es.close(); if (teachES === es) teachES = null; setTeachActive(false); };
  es.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.event === "progress") teachLog(`  ${m.message}`);
    else if (m.event === "complete") {
      stop(); $("teachBar").style.width = "100%";
      $("teachStatus").textContent = "Done";
      teachLog("Done.");
      renderTeach(m.result || {}, m.png);
      document.querySelector('[data-tab="teach"]').classList.add("done");
      toast("Leakage demo complete.", "ok");
    } else if (m.event === "error") {
      stop(); $("teachStatus").textContent = "";
      $("teachError").innerHTML = `<div class="notices">${notice("err", `<b>The demo stopped.</b> ${esc(m.error || "Unknown error.")}`)}</div>`;
      toast("The demo stopped with an error.", "err");
    }
  };
  es.onerror = () => {
    if (teachES !== es) return;
    stop(); $("teachStatus").textContent = "";
    $("teachError").innerHTML = `<div class="notices">${notice("err", "Lost the connection to the server while the demo was running. Is python app.py still running?")}</div>`;
    toast("Demo stream error.", "err");
  };
}
// {mean, min, max} -> "0.812 (0.80 to 0.83)" when the repeats differ
function rangeHtml(s, signed) {
  if (!s || !isNum(s.mean)) return "n/a";
  const f = (v) => (signed && v > 0 ? "+" : "") + v.toFixed(3);
  const spread = isNum(s.min) && isNum(s.max) && s.max - s.min >= 0.0005;
  return `${esc(f(s.mean))}${spread ? `<div class="ci small">${esc(f(s.min))} to ${esc(f(s.max))}</div>` : ""}`;
}
function renderTeach(r, png) {
  TEACH_RESULT = r;
  $("teachResults").style.display = "block";
  $("teachNotes").innerHTML = (r.notes || []).length ? r.notes.map(x => notice("info", esc(x))).join("") : "";
  const cards = [["Patients used", fmtInt(r.n_used)], ["Outcome rate", isNum(r.prevalence) ? `${fmtPct(r.prevalence)}%` : "n/a"]];
  if (isNum(r.chance_margin)) cards.push([term("Chance margin"), `±${r.chance_margin.toFixed(3)}`]);
  if (isNum(r.runtime_s)) cards.push(["Run time", `${Math.round(r.runtime_s)} s`]);
  $("teachCards").innerHTML = cards.map(([l, v]) => `<div class="card"><div class="lbl">${l}</div><div class="big">${esc(v)}</div></div>`).join("");
  const t = $("teachTable");
  t.innerHTML = `<tr><th>Approach</th><th class="nw">Reported AUROC</th><th class="nw">Honest ${term("lockbox", "Lockbox")} AUROC</th>
    <th>${term("Optimism")}</th><th class="nw">${term("Calibration slope", "CalibrationSlope")}</th><th>${term("CITL", "CalibrationInTheLarge")}</th><th>Reported on</th></tr>`;
  (r.arms || []).forEach(a => {
    const good = a.arm === "correct";
    const flag = a.optimism && isNum(a.optimism.mean) && a.optimism.mean >= 0.05;
    const tr = el("tr", { className: good ? "row-good" : "" });
    tr.innerHTML = `<td style="min-width:230px;max-width:320px"><details style="margin:0;border:0;padding:0;background:none">
        <summary style="font-size:13px;color:var(--ink);line-height:1.4">${esc(a.label)}${good ? ` <span class="tag ok">correct</span>` : ""}${flag ? ` <span class="tag warn">flatters</span>` : ""}</summary>
        <div class="small muted" style="margin-top:6px;line-height:1.55;font-weight:400">${esc(a.explanation || "")}</div></details></td>
      <td class="nw">${rangeHtml(a.reported_auroc)}</td><td class="nw">${rangeHtml(a.lockbox_auroc)}</td>
      <td class="nw${flag ? " flag" : ""}">${rangeHtml(a.optimism, true)}</td>
      <td class="nw">${rangeHtml(a.calibration_slope)}</td><td class="nw">${rangeHtml(a.citl, true)}</td>
      <td class="small" style="min-width:150px">${esc(a.reported_on || "")}</td>`;
    t.appendChild(tr);
  });
  const pw = $("teachPlot"); pw.innerHTML = "";
  if (png) {
    pw.innerHTML = `<img class="plot" src="${png}" style="max-width:760px"><div class="flex" style="margin-top:6px"></div>`;
    const b = pw.querySelector(".flex");
    b.appendChild(dlButton("PNG", () => dlPng(png, "leakage_demo.png")));
    b.appendChild(dlButton("PDF", () => dlPdf("teaching", { result: r }, "leakage_demo")));
  }
  $("teachResults").scrollIntoView({ behavior: "smooth", block: "start" });
}
$("teachCsvBtn").addEventListener("click", () => {
  const r = TEACH_RESULT; if (!r) return;
  const stats = ["reported_auroc", "lockbox_auroc", "optimism", "calibration_slope", "citl"];
  const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const head = ["arm", "label", "reported_on", ...stats.flatMap(s => [`${s}_mean`, `${s}_min`, `${s}_max`]), "explanation"];
  const lines = (r.arms || []).map(a => [a.arm, a.label, a.reported_on,
    ...stats.flatMap(s => [(a[s] || {}).mean, (a[s] || {}).min, (a[s] || {}).max]), a.explanation].map(cell).join(","));
  dlBlob(head.map(cell).join(",") + "\n" + lines.join("\n"), "leakage_demo.csv");
});

// =====================================================================
// 7. reports
// =====================================================================
async function initReports() {
  const d = await jget("/api/session");
  const confirmed = d.active && d.meta && d.meta.stage === "confirmed";
  $("reportsEmpty").style.display = confirmed ? "none" : "block";
  $("reportsBody").style.display = confirmed ? "block" : "none";
  if (!confirmed) return;
  loadMethodsText(); loadTripodReport(); loadLogCheck();
}
async function loadLogCheck() {
  const d = await jget("/api/report/verify-log");
  const box = $("logIntegrity");
  if (d.error) { box.textContent = ""; return; }
  if (d.ok) {
    box.innerHTML = `<span class="ok-text">Analysis log intact.</span> ${d.n_entries} entr${d.n_entries === 1 ? "y" : "ies"}, ` +
      `each linked to the one before by a SHA-256 hash.` +
      (d.legacy ? " Entries written before this check existed cannot be verified." : "");
  } else {
    box.innerHTML = `<span class="err-text">Analysis log check failed at line ${esc(d.first_bad_line)}.</span> ` +
      `The log was edited or damaged after it was written, so the reports below may not reflect what was run.`;
  }
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
  if (!editing) { p.focus(); toast("Editing locally, not saved to the server.", ""); }
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
  $("tripodDisclaimer").innerHTML = `<span class="disc-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9 16H3L12 3z"/><line x1="12" y1="10" x2="12" y2="14"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span><span class="disc-text"></span>`;
  $("tripodDisclaimer").querySelector(".disc-text").textContent = d.disclaimer;
  const s = d.summary;
  $("tripodSummary").innerHTML = [
    ["Found", s.found], ["Not found", s.not_found],
    ["Not applicable", s.not_applicable], ["Manual review", s.manual_review]
  ].map(([l, v]) => `<div class="card"><div class="lbl">${l}</div><div class="big">${v}</div></div>`).join("");
  const t = $("tripodTable"); t.innerHTML = "";
  t.appendChild(el("tr", {}, `<th>Section</th><th>Item</th><th>Status</th>`));
  d.items.forEach(i => t.appendChild(el("tr", {}, `<td class="small">${esc(i.section)}</td>
    <td class="small">${esc(i.description)}${i.note ? `<div class="muted" style="margin-top:3px">${esc(i.note)}</div>` : ""}</td>
    <td><span class="pill ${esc(i.status)}">${esc(i.status.replace(/_/g, " "))}</span></td>`)));
}

// =====================================================================
// downloads
// =====================================================================
function dlButton(label, onClick) {
  const b = el("button", { className: "ghost sm", textContent: label });
  b.onclick = onClick;
  return b;
}
function dlPng(dataUrl, fname) { const a = el("a", { href: dataUrl, download: fname }); a.click(); }
// any chart the server knows how to draw, as a pdf (/api/render/pdf)
async function dlPdf(kind, data, name) {
  let r;
  try {
    r = await fetch("/api/render/pdf", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, data, name }) });
  } catch (e) { toast("Cannot reach the server. Is python app.py still running?", "err"); return; }
  const type = r.headers.get("Content-Type") || "";
  if (!r.ok || !type.includes("pdf")) {
    let msg = null;
    try { msg = (await r.json()).error; } catch (e) { /* not json */ }
    toast(msg || `Could not make the PDF (HTTP ${r.status}).`, "err"); return;
  }
  const url = URL.createObjectURL(await r.blob());
  const a = el("a", { href: url, download: `${name}.pdf` }); a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
function dlBlob(text, fname, type = "text/csv") {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: fname }); a.click();
}
function tableToCsv(table) {
  return [...table.rows].map(r => [...r.cells].map(c =>
    `"${c.textContent.replace(/"/g, '""').trim()}"`).join(",")).join("\n");
}
function dlTable(id, fname) { dlBlob(tableToCsv($(id)), fname); }
function dlImpCsv(rows, fname) {
  const keys = Object.keys(rows[0] || { feature: "", importance: "" });
  const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;     // feature names can hold commas
  const csv = keys.join(",") + "\n" + rows.map(r => keys.map(k => cell(r[k])).join(",")).join("\n");
  dlBlob(csv, fname);
}

// =====================================================================
// dataset history (sidebar)
// =====================================================================
function applySessionMeta(m) {
  if (!m) { $("sessionTag").textContent = "No dataset loaded"; State.confirmed = false; State.cohort = null; return; }
  $("sessionTag").innerHTML = `Dataset: <b style="color:#fff">${esc(m.filename || "loaded")}</b><br>${m.n_rows || ""} rows`;
  State.confirmed = m.stage === "confirmed";
  State.coltypes = m.coltypes || {};
  // a confirmed session can carry its cohort flow in meta
  State.cohort = m.cohort && m.cohort.flow ? m.cohort : null;
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
    item.innerHTML = `<span class="ds-label" title="${esc(label)}">${esc(label)}${s.n_rows ? ` · ${s.n_rows} rows` : ""}</span>
      <span class="ds-x" title="Delete dataset">&times;</span>`;
    item.querySelector(".ds-label").addEventListener("click", () => switchDataset(s.session_id));
    item.querySelector(".ds-x").addEventListener("click", (e) => { e.stopPropagation(); deleteDataset(s.session_id, label); });
    list.appendChild(item);
  });
}

async function switchDataset(sid) {
  const d = await jpost(`/api/sessions/${sid}/activate`, {});
  if (d.error) { toast(d.error, "err"); return; }
  applySessionMeta(d.meta); dataChanged();
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
  const r = await jdelete(`/api/sessions/${encodeURIComponent(sid)}`);
  if (r.error) { toast(r.error, "err"); return; }
  const sess = await jget("/api/session");
  if (!sess.active) applySessionMeta(null);
  dataChanged();
  loadDatasetList();
  toast("Dataset deleted.", "ok");
}

function resetUploadTab() {
  State.columns = []; State.coltypes = {}; State.confirmed = false; State.lastTrain = null;
  fileInput.value = "";
  $("uploadInfo").textContent = "";
  $("previewPanel").style.display = "none";
  $("typePanel").style.display = "none"; $("splitPanel").style.display = "none";
  $("previewTable").innerHTML = ""; $("typeTable").innerHTML = "";
  $("confirmMsg").textContent = "";
  $("smoteBox").style.display = "none";
  $("imbalanceNote").style.display = "none";
  $("stratCol").innerHTML = `<option value="">None</option>`;
  $("cohortPanel").style.display = "none";
  resetCohort();
  $("cohortSummaryPanel").style.display = "none";
}

$("clearAllBtn").addEventListener("click", async () => {
  const ok = await showConfirm("Clear all data?",
    "This permanently deletes every uploaded dataset on this machine. This cannot be undone.", "Clear all data");
  if (!ok) return;
  const r = await jdelete("/api/sessions");
  if (r.error) { toast(r.error, "err"); return; }
  applySessionMeta(null); dataChanged();
  resetUploadTab();
  document.querySelectorAll(".nav a.done").forEach(a => a.classList.remove("done"));
  loadDatasetList();
  document.querySelector('[data-tab="overview"]').click();
  toast("All data cleared.", "ok");
});

// =====================================================================
// drag-to-select for checklist grids (model pick/delete, model training list).
// click-to-toggle one at a time still works as before; this adds
// click-and-drag across a grid of checkboxes to select/deselect a whole
// run of them in one motion, so deleting a big batch of saved models
// doesnt mean clicking every single one.
// =====================================================================
function enableDragSelect(container) {
  if (!container || container._dragSelectBound) return;
  container._dragSelectBound = true;

  let dragging = false, dragValue = true, suppressClick = null;

  function checkboxFromNode(node) {
    const label = node && node.closest ? node.closest("label") : null;
    if (!label || !container.contains(label)) return null;
    return label.querySelector('input[type="checkbox"]');
  }
  function setChecked(cb, value) {
    suppressClick = cb;
    if (!cb || cb.disabled || cb.checked === value) return;
    cb.checked = value;
    cb.dispatchEvent(new Event("change", { bubbles: true }));
  }

  container.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const cb = checkboxFromNode(e.target);
    if (!cb) return;
    dragging = true;
    dragValue = !cb.checked;
    setChecked(cb, dragValue);
    container.classList.add("drag-active");
    document.body.style.userSelect = "none";
  });

  // we drive the checked state ourselves on mousedown/mousemove so a drag
  // can span many boxes in one gesure; without this, the browser's own
  // click-driven toggle right after would flip the last box back.
  container.addEventListener("click", (e) => {
    const cb = checkboxFromNode(e.target);
    if (cb && cb === suppressClick) { e.preventDefault(); suppressClick = null; }
  }, true);

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const cb = checkboxFromNode(document.elementFromPoint(e.clientX, e.clientY));
    if (cb) setChecked(cb, dragValue);
  });

  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    container.classList.remove("drag-active");
    document.body.style.userSelect = "";
  });
}
["modelList", "modelPickList"].forEach(id => enableDragSelect($(id)));

// restore session tag on load
jget("/api/session").then(d => applySessionMeta(d.active ? d.meta : null));
jget("/api/version").then(d => { if (d.version) $("appVersion").textContent = "· v" + d.version; });
loadDatasetList();
