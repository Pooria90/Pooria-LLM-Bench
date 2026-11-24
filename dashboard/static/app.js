"use strict";

// ---------------------------------------------------------------- state & utils

const state = {
  runs: [],
  runId: null,
  data: null,
  models: [],
  repeats: 1,
  samples: [],
  sampleById: new Map(),
  results: new Map(), // "model|sample|repeat" -> result
  filtered: [],
  selected: null,
  detailSig: null,
  timer: null,
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const key = (model, sample, rep) => `${model}|${sample}|${rep}`;
const num = (x, digits = 0) => (x == null || Number.isNaN(x) ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);
const money = (x, digits = 4) => (x == null ? "–" : `$${x.toFixed(digits)}`);
const secs = (x) => (x == null ? "–" : x < 60 ? `${x.toFixed(1)} s` : `${(x / 60).toFixed(1)} min`);
const shortName = (m) => m.split("/").pop();

async function api(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function readHash() {
  return new URLSearchParams(location.hash.slice(1));
}
function writeHash() {
  const h = new URLSearchParams();
  if (state.runId) h.set("run", state.runId);
  if (state.selected) h.set("sample", state.selected);
  history.replaceState(null, "", `#${h.toString()}`);
}

function wilson(k, n, z = 1.96) {
  if (!n) return null;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// ------------------------------------------------------------------- verdicts

function verdict(r) {
  if (!r) return { key: "pending", label: "Pending", glyph: "·" };
  if (r.error) return { key: "error", label: "Error", glyph: "!" };
  if (r.correct) return { key: "correct", label: "Correct", glyph: "✓" };
  const fr = r.finish_reasons || [];
  if (fr.includes("content_filter")) return { key: "refused", label: "Refused", glyph: "⊘" };
  if (fr.includes("length") || r.score_details?.empty_output === "length")
    return { key: "budget", label: "Out of budget", glyph: "✂" };
  return { key: "wrong", label: "Wrong", glyph: "✗" };
}
const noReasoning = (r) => r && !r.error && r.reasoning_tokens === 0;
const formatMiss = (r) =>
  r && !r.error && !r.score_details?.empty_output && (r.score_details?.format_ok ?? r.score_details?.answer_line) === false;

function resultsFor(sampleId) {
  const out = [];
  for (const m of state.models) for (let rep = 0; rep < state.repeats; rep++) {
    const r = state.results.get(key(m, sampleId, rep));
    if (r) out.push(r);
  }
  return out;
}

// --------------------------------------------------------------- math/markdown

const MATH_PATTERNS = [
  /\$\$[\s\S]+?\$\$/g,
  /\\\[[\s\S]+?\\\]/g,
  /\\\([\s\S]+?\\\)/g,
  /\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|alignat\*?)\}[\s\S]+?\\end\{\1\}/g,
  /\$(?=\S)(?:\\\$|[^$\n])+?(?<=\S)\$/g,
];
const DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "\\(", right: "\\)", display: false },
  ...["align", "equation", "gather", "multline", "alignat"].flatMap((env) => [
    { left: `\\begin{${env}*}`, right: `\\end{${env}*}`, display: true },
    { left: `\\begin{${env}}`, right: `\\end{${env}}`, display: true },
  ]),
  { left: "$", right: "$", display: false },
];
const MACROS = { "\\wideparen": "\\overgroup{#1}", "\\overarc": "\\overgroup{#1}" };

// Wrap a bare \boxed{...} (outside any math delimiters) so KaTeX renders it.
function wrapBareBoxed(text, stash) {
  let out = "";
  let i = 0;
  const re = /\\boxed\s*\{/g;
  let m;
  while ((m = re.exec(text))) {
    let depth = 1, j = m.index + m[0].length;
    while (j < text.length && depth) { depth += text[j] === "{" ? 1 : text[j] === "}" ? -1 : 0; j++; }
    if (depth) break;
    out += text.slice(i, m.index) + `XMATHX${stash.push(`\\(${text.slice(m.index, j)}\\)`) - 1}XMATHX`;
    i = j;
    re.lastIndex = j;
  }
  return out + text.slice(i);
}

function renderRich(el, text) {
  text = String(text ?? "");
  if (!window.marked || !window.DOMPurify) {
    el.innerHTML = `<pre class="raw">${esc(text)}</pre>`;
    return;
  }
  const stash = [];
  let t = text;
  for (const re of MATH_PATTERNS) t = t.replace(re, (m) => `XMATHX${stash.push(m) - 1}XMATHX`);
  t = wrapBareBoxed(t, stash);
  let html = DOMPurify.sanitize(marked.parse(t, { gfm: true, breaks: false }));
  html = html.replace(/XMATHX(\d+)XMATHX/g, (_, i) => esc(stash[Number(i)]));
  el.classList.add("rich");
  el.innerHTML = html;
  if (window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: DELIMITERS, throwOnError: false, strict: "ignore", macros: { ...MACROS },
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
    });
  }
}

// Figures given to the model as text: LaTeX environments, BBCode-style [asy], or an ASCII grid
// (3+ consecutive lines that start and end with "|").
const FIGURE_RE = /\\begin\{(tikzpicture|asy)\}[\s\S]*?\\end\{\1\}|\[asy\][\s\S]*?\[\/asy\]|(?:^\|.*\|[ \t]*(?:\n|$)){3,}/gm;
function renderProblem(el, text) {
  el.innerHTML = "";
  let last = 0;
  const appendRich = (chunk) => {
    if (!chunk.trim()) return;
    const div = document.createElement("div");
    renderRich(div, chunk);
    el.appendChild(div);
  };
  for (const m of String(text).matchAll(FIGURE_RE)) {
    appendRich(text.slice(last, m.index));
    if (m[0].startsWith("|")) {
      // ASCII diagram: it *is* the figure, so show it as-is (Markdown would collapse it).
      const d = document.createElement("div");
      d.className = "figure";
      d.innerHTML = `<p class="muted">Diagram, given to the model as this ASCII grid:</p><pre><code>${esc(m[0].replace(/\n$/, ""))}</code></pre>`;
      el.appendChild(d);
    } else {
      const kind = m[1] === "tikzpicture" ? "TikZ" : "Asymptote";
      const d = document.createElement("details");
      d.className = "figure";
      d.innerHTML = `<summary>Figure: given to the model as ${kind} source (show)</summary><pre><code>${esc(m[0])}</code></pre>`;
      el.appendChild(d);
    }
    last = m.index + m[0].length;
  }
  appendRich(text.slice(last));
}

// ---------------------------------------------------------------- data loading

async function loadRuns() {
  state.runs = await api("/api/runs");
  const sel = $("#run-select");
  sel.innerHTML = state.runs
    .map((r) => `<option value="${esc(r.id)}">${esc(r.id)} · ${esc(r.task ?? "?")} · ${r.n_results} results</option>`)
    .join("");
}

async function loadRunData() {
  const data = await api(`/api/run?id=${encodeURIComponent(state.runId)}`);
  state.data = data;
  const man = data.manifest || {};
  const invocations = [man, ...(man.resumed || [])];
  const models = [];
  for (const inv of invocations) for (const m of Object.keys(inv.models || {})) if (!models.includes(m)) models.push(m);
  for (const r of data.results) if (!models.includes(r.model)) models.push(r.model);
  state.models = models;
  state.repeats = Math.max(1, ...invocations.map((i) => i.config?.repeats || 1), ...data.results.map((r) => (r.repeat ?? 0) + 1));

  state.results = new Map(data.results.map((r) => [key(r.model, r.sample_id, r.repeat ?? 0), r]));
  let samples = data.samples || [];
  if (!samples.length) {
    // runs created before samples.jsonl existed: rebuild what we can from results
    const seen = new Map();
    for (const r of data.results) if (!seen.has(r.sample_id))
      seen.set(r.sample_id, { id: r.sample_id, input: "(problem text not saved for this run)", target: r.target, metadata: { category: r.category } });
    samples = [...seen.values()];
  }
  state.samples = samples;
  state.sampleById = new Map(samples.map((s) => [s.id, s]));
}

async function selectRun(runId, sampleId) {
  state.runId = runId;
  $("#run-select").value = runId;
  state.detailSig = null;
  await loadRunData();
  fillCategoryFilter();
  renderAll();
  const first = sampleId && state.sampleById.has(sampleId) ? sampleId : state.filtered[0];
  if (first) selectSample(first, false);
  else $("#detail").innerHTML = `<p class="muted">No samples in this run.</p>`;
  const expected = state.samples.length * state.models.length * state.repeats;
  $("#auto-refresh").checked = state.data.results.length < expected;
  syncTimer();
}

async function reload() {
  try {
    await loadRunData();
    renderAll();
    if (state.selected) {
      const sig = JSON.stringify(resultsFor(state.selected).map((r) => [r.model, r.repeat, verdict(r).key, (r.output || "").length]));
      if (sig !== state.detailSig) renderDetail(state.selected);
    }
  } catch (err) {
    console.error(err);
  }
}

function syncTimer() {
  clearInterval(state.timer);
  state.timer = $("#auto-refresh").checked ? setInterval(reload, 15000) : null;
}

// ------------------------------------------------------------------- rendering

function renderAll() {
  renderMeta();
  renderSummary();
  applyFilters();
  renderMatrix();
}

function renderMeta() {
  const man = state.data.manifest || {};
  const expected = state.samples.length * state.models.length * state.repeats;
  const done = state.data.results.length;
  const parts = [
    `<b>${esc(man.task ?? "?")}</b>`,
    `${state.samples.length} problems × ${state.models.length} models${state.repeats > 1 ? ` × ${state.repeats} repeats` : ""}`,
    `${done}/${expected} done`,
    man.created_at ? `started ${new Date(man.created_at).toLocaleString()}` : "",
    man.railtracks_version ? `railtracks ${esc(man.railtracks_version)}` : "",
    man.git_sha ? `git ${esc(man.git_sha)}` : "",
  ].filter(Boolean);
  $("#run-meta").innerHTML = parts.join(" · ");
}

function modelStats(model) {
  const rs = state.data.results.filter((r) => r.model === model);
  const scored = rs.filter((r) => !r.error && r.correct != null);
  const correct = scored.filter((r) => r.correct).length;
  const costs = rs.map((r) => r.cost_usd).filter((c) => c != null);
  const cost = costs.length ? costs.reduce((a, b) => a + b, 0) : null;
  const mean = (f) => (rs.length ? rs.reduce((a, r) => a + (f(r) || 0), 0) / rs.length : null);
  return {
    n: scored.length, correct, acc: scored.length ? correct / scored.length : null, ci: wilson(correct, scored.length),
    budget: rs.filter((r) => verdict(r).key === "budget").length,
    refused: rs.filter((r) => verdict(r).key === "refused").length,
    errors: rs.filter((r) => r.error).length,
    noThink: scored.filter(noReasoning).length,
    fmt: scored.filter(formatMiss).length,
    out: mean((r) => r.output_tokens), reasoning: mean((r) => r.reasoning_tokens),
    p50: median(scored.map((r) => r.latency_s)), cost, perCorrect: cost != null && correct ? cost / correct : null,
  };
}

function renderSummary() {
  const rows = state.models.map((m) => {
    const s = modelStats(m);
    return `<tr>
      <td title="${esc(m)}">${esc(m)}</td>
      <td class="num">${s.correct}/${s.n}</td>
      <td class="num"><b>${pct(s.acc)}</b></td>
      <td class="num">${s.ci ? `${(s.ci[0] * 100).toFixed(0)}–${(s.ci[1] * 100).toFixed(0)}%` : "–"}</td>
      <td class="num">${s.budget}</td>
      <td class="num">${s.refused}</td>
      <td class="num">${s.errors}</td>
      <td class="num">${s.noThink}</td>
      <td class="num">${s.fmt}</td>
      <td class="num">${num(s.out)}</td>
      <td class="num">${num(s.reasoning)}</td>
      <td class="num">${secs(s.p50)}</td>
      <td class="num">${money(s.cost, 2)}</td>
      <td class="num">${money(s.perCorrect, 3)}</td>
    </tr>`;
  });
  $("#summary").innerHTML = `<table class="summary">
    <thead><tr>
      <th>Model</th><th class="num">Correct</th><th class="num">Accuracy</th><th class="num" title="95% Wilson interval">95% CI</th>
      <th class="num" title="Ran out of output budget">✂</th><th class="num" title="Refused">⊘</th><th class="num" title="Errors (excluded from accuracy)">!</th>
      <th class="num" title="Answered with zero reasoning tokens">0r</th><th class="num" title="Answer not in \\boxed{}">no-fmt</th>
      <th class="num">Avg out tok</th><th class="num">Avg reasoning</th><th class="num">Median time</th><th class="num">Cost</th><th class="num">$/correct</th>
    </tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function fillCategoryFilter() {
  const cats = [...new Set(state.samples.map((s) => s.metadata?.category).filter(Boolean))].sort();
  const types = [...new Set(state.samples.flatMap((s) => s.metadata?.problem_type || []))].sort();
  $("#filter-category").innerHTML =
    `<option value="">All categories</option>` +
    cats.map((c) => `<option value="cat:${esc(c)}">${esc(c)}</option>`).join("") +
    types.map((t) => `<option value="type:${esc(t)}">${esc(t)}</option>`).join("");
}

function applyFilters() {
  const outcome = $("#filter-outcome").value;
  const cat = $("#filter-category").value;
  const text = $("#filter-text").value.trim().toLowerCase();
  state.filtered = state.samples
    .filter((s) => {
      if (cat.startsWith("cat:") && s.metadata?.category !== cat.slice(4)) return false;
      if (cat.startsWith("type:") && !(s.metadata?.problem_type || []).includes(cat.slice(5))) return false;
      if (text && !s.id.toLowerCase().includes(text) && !String(s.input).toLowerCase().includes(text)) return false;
      const vs = resultsFor(s.id).map(verdict).map((v) => v.key);
      switch (outcome) {
        case "anywrong": return vs.some((k) => k !== "correct");
        case "disagree": return vs.includes("correct") && vs.some((k) => k !== "correct");
        case "allwrong": return vs.length > 0 && !vs.includes("correct");
        case "issues": return vs.some((k) => ["error", "budget", "refused"].includes(k));
        case "nothink": return resultsFor(s.id).some(noReasoning);
        default: return true;
      }
    })
    .map((s) => s.id);
}

function renderMatrix() {
  const hasTypes = state.samples.some((s) => (s.metadata?.problem_type || []).length);
  const head = state.models.map((m) => `<th class="num" title="${esc(m)}">${esc(shortName(m))}</th>`).join("");
  const rows = state.filtered.map((id) => {
    const s = state.sampleById.get(id);
    const cells = state.models.map((m) => {
      const glyphs = [];
      for (let rep = 0; rep < state.repeats; rep++) {
        const r = state.results.get(key(m, id, rep));
        const v = verdict(r);
        const tip = r ? `${v.label}${r.extracted != null ? ` · answer ${r.extracted}` : ""}` : "Pending";
        glyphs.push(`<span class="v v-${v.key}" title="${esc(tip)}">${v.glyph}</span>${noReasoning(r) ? `<span class="tag-mini" title="no reasoning tokens">0r</span>` : ""}`);
      }
      return `<td class="cell">${glyphs.join(" ")}</td>`;
    });
    const types = (s.metadata?.problem_type || []).join(", ");
    const fig = s.metadata?.figure ? ` <span class="badge" title="figure given as ${esc(s.metadata.figure)}">fig</span>` : "";
    return `<tr data-id="${esc(id)}" class="${id === state.selected ? "selected" : ""}">
      <td class="id">${esc(id)}${fig}</td>${hasTypes ? `<td class="ptype">${esc(types)}</td>` : ""}<td class="num">${esc(s.target)}</td>${cells.join("")}</tr>`;
  });
  $("#matrix").innerHTML = rows.length
    ? `<table class="matrix"><thead><tr><th>Problem</th>${hasTypes ? "<th>Type</th>" : ""}<th class="num">Answer</th>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`
    : `<p class="muted">No problems match these filters.</p>`;
}

function selectSample(id, scroll = true) {
  state.selected = id;
  writeHash();
  document.querySelectorAll(".matrix tr.selected").forEach((tr) => tr.classList.remove("selected"));
  const row = document.querySelector(`.matrix tr[data-id="${CSS.escape(id)}"]`);
  if (row) {
    row.classList.add("selected");
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }
  renderDetail(id);
}

function step(delta) {
  if (!state.filtered.length) return;
  const i = state.filtered.indexOf(state.selected);
  const next = i < 0 ? 0 : Math.min(state.filtered.length - 1, Math.max(0, i + delta));
  selectSample(state.filtered[next]);
}

function renderDetail(id) {
  const s = state.sampleById.get(id);
  if (!s) return;
  const pos = state.filtered.indexOf(id);
  const md = s.metadata || {};
  const badges = [
    md.category ? `<span class="badge">${esc(md.category)}</span>` : "",
    md.figure ? `<span class="badge">figure: ${esc(md.figure)}</span>` : "",
    ...(md.problem_type || []).map((t) => `<span class="badge">${esc(t)}</span>`),
  ].join(" ");
  const el = $("#detail");
  el.innerHTML = `
    <div class="detail-head">
      <button type="button" data-step="-1" title="Previous (← or k)">←</button>
      <span class="title">${esc(id)}</span>
      <button type="button" data-step="1" title="Next (→ or j)">→</button>
      <span class="muted">${pos >= 0 ? `${pos + 1} / ${state.filtered.length}` : "not in current filter"}</span>
      ${badges}
      <span class="spacer"></span>
      ${md.aops_url ? `<a href="${esc(md.aops_url)}" target="_blank" rel="noopener">Problem page ↗</a>` : ""}
    </div>
    <h3>Problem</h3>
    <div id="problem"></div>
    <div class="answer-key">Correct answer: <b>${esc(s.target)}</b></div>
    <h3>Model answers</h3>
    <div class="cards" id="cards"></div>`;
  renderProblem($("#problem"), s.input);
  const cards = $("#cards");
  for (const m of state.models) for (let rep = 0; rep < state.repeats; rep++) cards.appendChild(renderCard(m, id, rep));
  el.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => step(Number(b.dataset.step))));
  state.detailSig = JSON.stringify(resultsFor(id).map((r) => [r.model, r.repeat, verdict(r).key, (r.output || "").length]));
}

function renderCard(model, sampleId, rep) {
  const r = state.results.get(key(model, sampleId, rep));
  const v = verdict(r);
  const card = document.createElement("article");
  card.className = `card ${v.key}`;
  const name = `${model}${state.repeats > 1 ? ` · run ${rep + 1}` : ""}`;
  if (!r) {
    card.innerHTML = `<div class="card-head"><span class="model">${esc(name)}</span><span class="badge">Pending</span></div>`;
    return card;
  }
  const finish = [...new Set((r.finish_reasons || []).filter(Boolean))].join(", ") || "–";
  const flags = [
    noReasoning(r) ? `<span class="badge budget" title="The model produced no reasoning tokens">no reasoning</span>` : "",
    formatMiss(r) ? `<span class="badge" title="The answer was not given in \\boxed{}">no \\boxed{}</span>` : "",
    r.score_details?.empty_output ? `<span class="badge budget">no answer text (${esc(r.score_details.empty_output)})</span>` : "",
  ].filter(Boolean).join("");
  card.innerHTML = `
    <div class="card-head">
      <span class="model">${esc(name)}</span>
      <span class="badge ${v.key}">${v.label}</span>
      <span class="given">answer: <b>${esc(r.extracted ?? "–")}</b></span>
    </div>
    <div class="stats">
      <span>${secs(r.latency_s)}</span>
      <span>${num(r.output_tokens)} out tok</span>
      <span>${num(r.reasoning_tokens)} reasoning</span>
      <span>${num(r.input_tokens)} in tok</span>
      <span>${money(r.cost_usd)}</span>
      <span>finish: ${esc(finish)}</span>
      ${r.n_calls > 1 ? `<span>${r.n_calls} calls</span>` : ""}
    </div>
    ${flags ? `<div class="flags">${flags}</div>` : ""}
    ${r.error ? `<div class="error-text">${esc(r.error)}</div>` : ""}
    <div class="output"></div>
    <div class="card-actions">
      <button type="button" data-act="expand">Expand answer</button>
      <button type="button" data-act="reasoning">Show reasoning</button>
      <button type="button" data-act="prompt">Show exact prompt</button>
      <button type="button" data-act="raw">Show raw answer text</button>
    </div>
    <div class="lazy"></div>`;
  const out = card.querySelector(".output");
  if (r.output != null && String(r.output).trim()) renderRich(out, r.output);
  else out.innerHTML = `<p class="muted">No answer text was returned.</p>`;
  card.querySelector(".card-actions").addEventListener("click", (e) => {
    const act = e.target?.dataset?.act;
    if (act) onCardAction(act, card, r, e.target);
  });
  return card;
}

async function onCardAction(act, card, r, button) {
  const lazy = card.querySelector(".lazy");
  if (act === "expand") {
    const out = card.querySelector(".output");
    out.classList.toggle("expanded");
    button.textContent = out.classList.contains("expanded") ? "Collapse answer" : "Expand answer";
    return;
  }
  if (act === "raw") {
    lazy.innerHTML = `<h3>Raw answer text</h3><pre class="raw">${esc(r.output ?? "")}</pre>`;
    return;
  }
  lazy.innerHTML = `<p class="muted">Loading…</p>`;
  let t;
  try {
    t = await api(`/api/transcript?id=${encodeURIComponent(state.runId)}&model=${encodeURIComponent(r.model)}&sample=${encodeURIComponent(r.sample_id)}&repeat=${r.repeat ?? 0}`);
  } catch {
    lazy.innerHTML = `<p class="muted">No transcript saved for this answer.</p>`;
    return;
  }
  const messages = (t.histories || []).flatMap((h) => h.messages || []);
  if (act === "reasoning") {
    const parts = messages.filter((m) => m.role === "assistant" && m.reasoning).map((m) => m.reasoning);
    lazy.innerHTML = parts.length
      ? `<h3>Reasoning returned by the provider (${num(r.reasoning_tokens)} reasoning tokens billed)</h3><pre class="raw">${esc(parts.join("\n\n---\n\n"))}</pre>`
      : `<p class="muted">The provider returned no reasoning text (${num(r.reasoning_tokens)} reasoning tokens were used). OpenAI keeps reasoning hidden; Anthropic returns summaries; Radium returns the full trace.</p>`;
  } else if (act === "prompt") {
    const shown = messages.filter((m) => m.role === "system" || m.role === "user");
    lazy.innerHTML = shown.length
      ? `<h3>Exact prompt sent</h3>${shown.map((m) => `<p class="muted">${esc(m.role)}</p><pre class="raw">${esc(m.content)}</pre>`).join("")}`
      : `<p class="muted">No prompt messages in the transcript.</p>`;
  }
}

// ------------------------------------------------------------------- wiring

function bindUI() {
  $("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
  $("#refresh").addEventListener("click", async () => { await loadRuns(); $("#run-select").value = state.runId; await reload(); });
  $("#auto-refresh").addEventListener("change", syncTimer);
  for (const id of ["#filter-outcome", "#filter-category"]) $(id).addEventListener("change", () => { applyFilters(); renderMatrix(); });
  $("#filter-text").addEventListener("input", () => { applyFilters(); renderMatrix(); });
  $("#matrix").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) selectSample(tr.dataset.id, false);
  });
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowRight" || e.key === "j") { step(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft" || e.key === "k") { step(-1); e.preventDefault(); }
    else if (e.key === "r") reload();
  });
}

async function init() {
  bindUI();
  try {
    await loadRuns();
  } catch (err) {
    $("#detail").innerHTML = `<p class="error-text">Could not reach the dashboard server: ${esc(err.message)}</p>`;
    return;
  }
  if (!state.runs.length) {
    $("#detail").innerHTML = `<p class="muted">No runs found. Run a benchmark first (results go under runs/).</p>`;
    return;
  }
  const h = readHash();
  const wanted = h.get("run");
  const runId = wanted && state.runs.some((r) => r.id === wanted) ? wanted : state.runs[0].id;
  await selectRun(runId, h.get("sample"));
}

window.addEventListener("load", init);
