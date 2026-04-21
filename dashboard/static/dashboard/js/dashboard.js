/* BekaaSense dashboard - FIXED v3 */
const $ = (id) => document.getElementById(id);
let trendChart = null;
let shapChart = null;
let clfChart = null;
const ZONE_COLOR = {"Humid":"#2563eb","Sub-humid":"#22c55e","Semi-arid":"#eab308","Arid":"#f97316","Hyper-arid":"#dc2626"};
const MAX_HORIZON_MONTHS = 24;

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) { const body = await res.text().catch(() => ""); throw new Error(`${res.status} ${res.statusText}: ${body}`); }
  return res.json();
}
function toDate(y, m) { return new Date(y, m - 1, 15); }
function monthsBetween(fromY, fromM, toY, toM) { return (toY - fromY) * 12 + (toM - fromM); }
function parseMonthInput(val) { if (!val) return null; const [y, m] = val.split("-").map(Number); return { year: y, month: m }; }
function fmtYM(y, m) { return `${y}-${String(m).padStart(2, "0")}`; }

async function initMonthPicker() {
  const stations = await fetchJSON("/api/stations/");
  const s = (stations.stations || [])[0];
  if (!s) return null;
  const addMonths = (y, m, delta) => { const t = (y * 12 + (m - 1)) + delta; return { year: Math.floor(t / 12), month: (t % 12) + 1 }; };
  const minPick = addMonths(s.latest_year, s.latest_month, 1);
  const defPick = addMonths(s.latest_year, s.latest_month, 6);
  const maxPick = addMonths(s.latest_year, s.latest_month, MAX_HORIZON_MONTHS);
  const input = $("target_month");
  input.min = fmtYM(minPick.year, minPick.month);
  input.max = fmtYM(maxPick.year, maxPick.month);
  input.value = fmtYM(defPick.year, defPick.month);
}

async function renderTrend(station) {
  const data = await fetchJSON(`/api/trend/?station=${encodeURIComponent(station)}`);
  const hist = (data.history || []).map(p => ({ x: toDate(p.year, p.month), y: p.de_martonne, zone: p.aridity_zone }));
  const fc = (data.forecast || []).slice(0, MAX_HORIZON_MONTHS).map(p => ({ x: toDate(p.year, p.month), y: p.de_martonne_pred, lo: p.lower, hi: p.upper }));
  const hiBand = fc.map(p => ({ x: p.x, y: p.hi }));
  const loBand = fc.map(p => ({ x: p.x, y: p.lo }));
  const pc = hist.map(p => ZONE_COLOR[p.zone] || "#6b7280");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart($("trendChart").getContext("2d"), {
    type: "line",
    data: { datasets: [
      { label: "Observed De Martonne", data: hist, borderColor: "#556b2f", borderWidth: 2, pointRadius: 2.5, pointBackgroundColor: pc, pointBorderColor: pc, tension: 0.25, fill: false },
      { label: "Forecast", data: fc, borderColor: "#c1440e", borderDash: [6,4], borderWidth: 2, pointRadius: 0, tension: 0.25, fill: false },
      { label: "90% Upper", data: hiBand, borderColor: "rgba(193,68,14,0)", backgroundColor: "rgba(193,68,14,0.12)", fill: "+1", pointRadius: 0 },
      { label: "90% Lower", data: loBand, borderColor: "rgba(193,68,14,0)", pointRadius: 0 }
    ]},
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false },
      scales: { x: { type: "time", time: { unit: "year" }, title: { display: true, text: "Date" } }, y: { title: { display: true, text: "De Martonne Aridity Index" } } },
      plugins: { legend: { position: "bottom", labels: { filter: (it) => !it.text.includes("Upper") && !it.text.includes("Lower") } },
        tooltip: { callbacks: { afterLabel: (c) => { const p = c.raw; if (p.zone) return `Zone: ${p.zone}`; if (p.lo !== undefined) return `Interval: [${p.lo.toFixed(2)}, ${p.hi.toFixed(2)}]`; return ""; } } } }
    }
  });
}

function viabilityFromDM(dm) {
  if (dm >= 20) return { status: "green", label: "Viable", note: "Rain-fed wheat is climatically viable." };
  if (dm >= 10) return { status: "yellow", label: "Marginal", note: "Rain-fed wheat is marginal; irrigation strongly advised." };
  return { status: "red", label: "Non-viable", note: "Rain-fed wheat not climatically viable; irrigation required." };
}

async function renderViability(station, targetYM) {
  const stations = await fetchJSON("/api/stations/");
  const row = (stations.stations || []).find(s => s.station === station);
  if (row) {
    const v = row.crop_viability;
    $("viabilityDotNow").className = "dot " + v.status;
    $("viabilityLabelNow").textContent = `${v.label} - DM = ${row.latest_de_martonne.toFixed(2)}`;
    $("viabilityNoteNow").textContent = `${v.note} Latest observation: ${row.latest_year}-${String(row.latest_month).padStart(2,"0")} (${row.current_zone}).`;
  }
  if (!targetYM || !row) return;
  const h = monthsBetween(row.latest_year, row.latest_month, targetYM.year, targetYM.month);
  if (h <= 0) { $("forecastSubtitle").textContent = "Forecast month"; $("viabilityLabelFuture").textContent = "-"; $("viabilityNoteFuture").textContent = "Selected month is at or before the latest observation."; $("viabilityDotFuture").className = "dot"; return; }
  if (h > MAX_HORIZON_MONTHS) { $("forecastSubtitle").textContent = `${targetYM.year}-${String(targetYM.month).padStart(2,"0")} (out of range)`; $("viabilityLabelFuture").textContent = "Beyond supported horizon"; $("viabilityNoteFuture").textContent = `BekaaSense is validated only up to ${MAX_HORIZON_MONTHS} months ahead.`; $("viabilityDotFuture").className = "dot"; return; }
  try {
    const fc = await fetchJSON("/api/predict/", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ station, horizon_months: h, alpha: 0.1 }) });
    const t = (fc.forecast || []).slice(-1)[0];
    if (!t) return;
    const v = viabilityFromDM(t.de_martonne_pred);
    $("forecastSubtitle").textContent = `${t.year}-${String(t.month).padStart(2,"0")} - horizon +${h}mo`;
    $("viabilityDotFuture").className = "dot " + v.status;
    $("viabilityLabelFuture").textContent = `${v.label} - DM = ${t.de_martonne_pred.toFixed(2)}`;
    $("viabilityNoteFuture").textContent = `${v.note} Projected zone: ${t.aridity_zone}. 90% interval: [${t.lower.toFixed(1)}, ${t.upper.toFixed(1)}].`;
  } catch (e) { console.warn(e); }
}

async function renderSHAP(station) {
  let data;
  try { data = await fetchJSON("/api/explain/", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ station, top_k: 8 }) }); } catch (e) { console.warn(e); return; }
  const f = data.top_features || [];
  const labels = f.map(x => x.feature); const values = f.map(x => x.shap);
  const colors = values.map(v => v >= 0 ? "rgba(34,197,94,0.7)" : "rgba(220,38,38,0.7)");
  if (shapChart) shapChart.destroy();
  shapChart = new Chart($("shapChart").getContext("2d"), {
    type: "bar", data: { labels, datasets: [{ data: values, backgroundColor: colors, borderColor: colors.map(c => c.replace("0.7","1")), borderWidth: 1 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
      scales: { x: { title: { display: true, text: "Contribution to predicted index" } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => { const x = f[c.dataIndex]; return `value=${x.value.toFixed(3)}  shap=${x.shap.toFixed(3)}`; } } } } }
  });
}

async function renderClassifier(station) {
  let data;
  try { data = await fetchJSON(`/api/latest_zone/?station=${encodeURIComponent(station)}`); } catch (e) { console.warn(e); return; }
  $("clfZone").textContent = data.classifier_prediction;
  const ag = $("clfAgreement");
  if (data.agreement) { ag.innerHTML = `Agrees with regressor (<strong>${data.regressor_zone}</strong>) for ${data.year}-${String(data.month).padStart(2,"0")}.`; ag.style.color = "#22c55e"; }
  else { ag.innerHTML = `Disagrees - regressor: <strong>${data.regressor_zone}</strong>, classifier: <strong>${data.classifier_prediction}</strong>.`; ag.style.color = "#dc2626"; }
  const labels = data.class_probabilities.map(c => c.zone);
  const probs = data.class_probabilities.map(c => c.probability);
  const colors = labels.map(z => ZONE_COLOR[z] || "#6b7280");
  if (clfChart) clfChart.destroy();
  clfChart = new Chart($("clfChart").getContext("2d"), {
    type: "bar", data: { labels, datasets: [{ data: probs, backgroundColor: colors }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
      scales: { x: { min: 0, max: 1, title: { display: true, text: "Probability" } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `p = ${c.parsed.x.toFixed(3)}` } } } }
  });
}

async function renderLeaderboard() {
  let data;
  try { data = await fetchJSON("/api/leaderboard/"); }
  catch (e) { $("lbTable").querySelector("tbody").innerHTML = '<tr><td colspan="8" style="text-align:center;color:#6b7280;">Run python -m model_engine.train</td></tr>'; return; }
  const rows = data.leaderboard || [];
  const tbody = $("lbTable").querySelector("tbody"); tbody.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");
    const cov = r.interval_coverage_90 !== undefined ? (r.interval_coverage_90 * 100).toFixed(1) + "%" : "-";
    const covStyle = r.interval_coverage_90 !== undefined && r.interval_coverage_90 < 0.88 ? 'style="color:#f97316;font-weight:600;"' : "";
    tr.innerHTML = `<td><strong>${r.model}</strong></td><td>${r.task}</td>` +
      `<td>${r.rmse !== undefined ? r.rmse.toFixed(3) : "-"}</td>` +
      `<td>${r.mae !== undefined ? r.mae.toFixed(3) : "-"}</td>` +
      `<td>${r.r2 !== undefined ? r.r2.toFixed(3) : "-"}</td>` +
      `<td>${r.bias !== undefined ? r.bias.toFixed(3) : "-"}</td>` +
      `<td ${covStyle}>${cov}</td>` +
      `<td>${r.f1_weighted !== undefined ? r.f1_weighted.toFixed(3) : "-"}</td>`;
    tbody.appendChild(tr);
  });
}

async function renderScoringMetrics() {
  let data;
  try { data = await fetchJSON("/api/scoring/"); }
  catch (e) { return; }

  // Health badges
  const health = data.model_health || {};
  const badgesEl = $("healthBadges");
  if (badgesEl) {
    badgesEl.innerHTML = "";
    const MODELS = ["RandomForest", "XGBoost", "AridityZoneClassifier"];
    MODELS.forEach(name => {
      const h = health[name];
      if (!h) return;
      const pass = h.pass;
      const badge = document.createElement("div");
      badge.style.cssText = `padding:8px 14px;border-radius:6px;font-size:0.85rem;font-weight:600;background:${pass?"#dcfce7":"#fee2e2"};color:${pass?"#166534":"#991b1b"};border:1px solid ${pass?"#86efac":"#fca5a5"}`;
      let detail = "";
      if (h.r2 !== undefined) detail = `R²=${h.r2.toFixed(3)}, RMSE=${h.rmse.toFixed(2)}, Cov=${(h.interval_coverage_90*100).toFixed(1)}%`;
      else if (h.f1_weighted !== undefined) detail = `F1w=${h.f1_weighted.toFixed(3)}, F1m=${h.f1_macro.toFixed(3)}`;
      badge.innerHTML = `${pass ? "✓" : "✗"} ${name}<br><span style="font-weight:400;font-size:0.78rem;">${detail}</span>`;
      badgesEl.appendChild(badge);
    });
    if (health.retrain_recommended) {
      const warn = document.createElement("div");
      warn.style.cssText = "padding:8px 14px;border-radius:6px;font-size:0.85rem;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;font-weight:600;";
      warn.textContent = "Retraining recommended — run: make train";
      badgesEl.appendChild(warn);
    }
  }

  // Per-class classifier table
  const cr = data.classifier_report;
  const ctbody = $("classifierTable") && $("classifierTable").querySelector("tbody");
  if (cr && ctbody) {
    ctbody.innerHTML = "";
    const SKIP = ["accuracy", "macro avg", "weighted avg"];
    Object.entries(cr.per_class || {}).forEach(([cls, m]) => {
      if (SKIP.includes(cls)) return;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><strong>${cls}</strong></td>` +
        `<td>${m.precision !== undefined ? m.precision.toFixed(3) : "-"}</td>` +
        `<td>${m.recall !== undefined ? m.recall.toFixed(3) : "-"}</td>` +
        `<td>${m["f1-score"] !== undefined ? m["f1-score"].toFixed(3) : "-"}</td>` +
        `<td>${m.support !== undefined ? m.support : "-"}</td>`;
      ctbody.appendChild(tr);
    });
    // Summary rows
    ["macro avg", "weighted avg"].forEach(key => {
      const m = cr.per_class[key];
      if (!m) return;
      const tr = document.createElement("tr");
      tr.style.fontStyle = "italic"; tr.style.color = "#6b7280";
      tr.innerHTML = `<td>${key}</td>` +
        `<td>${m.precision !== undefined ? m.precision.toFixed(3) : "-"}</td>` +
        `<td>${m.recall !== undefined ? m.recall.toFixed(3) : "-"}</td>` +
        `<td>${m["f1-score"] !== undefined ? m["f1-score"].toFixed(3) : "-"}</td>` +
        `<td>${m.support !== undefined ? m.support : "-"}</td>`;
      ctbody.appendChild(tr);
    });
  }

  // Confusion matrix
  const cmEl = $("confusionMatrix");
  if (cr && cmEl && cr.confusion_matrix && cr.labels) {
    const labels = cr.labels;
    const matrix = cr.confusion_matrix;
    let html = '<table style="border-collapse:collapse;font-size:0.82rem;">';
    html += '<thead><tr><th style="padding:4px 8px;background:#f3f4f6;border:1px solid #e5e7eb;"></th>';
    labels.forEach(l => { html += `<th style="padding:4px 8px;background:#f3f4f6;border:1px solid #e5e7eb;white-space:nowrap;">Pred: ${l}</th>`; });
    html += "</tr></thead><tbody>";
    matrix.forEach((row, i) => {
      html += `<tr><td style="padding:4px 8px;background:#f3f4f6;border:1px solid #e5e7eb;font-weight:600;white-space:nowrap;">True: ${labels[i]}</td>`;
      row.forEach((val, j) => {
        const onDiag = i === j;
        const bg = onDiag ? "#dcfce7" : (val > 0 ? "#fee2e2" : "#fff");
        html += `<td style="padding:4px 12px;border:1px solid #e5e7eb;text-align:center;background:${bg};font-weight:${onDiag?"600":"400"}">${val}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    cmEl.innerHTML = html;
  }
}

async function refresh() {
  const station = $("station").value;
  const tmEl = $("target_month");
  const targetYM = tmEl && tmEl.value ? parseMonthInput(tmEl.value) : null;
  try { await Promise.all([renderTrend(station), renderViability(station, targetYM), renderSHAP(station), renderClassifier(station)]); }
  catch (e) { console.error(e); }
}

document.addEventListener("DOMContentLoaded", async () => {
  await initMonthPicker();
  $("run").addEventListener("click", refresh);
  $("station").addEventListener("change", async () => { await initMonthPicker(); refresh(); });
  $("target_month").addEventListener("change", refresh);
  renderLeaderboard();
  renderScoringMetrics();
  refresh();
});
