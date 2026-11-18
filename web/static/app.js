"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  payload: null,
  enumeration: null,
};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  buildExampleButtons();
  $("#sql").value = window.EXAMPLES[0].sql;
  $("#btn-run").addEventListener("click", run);
  $("#btn-brute").addEventListener("click", runBruteForce);
  $("#sql").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
  });
  loadEnumeration();
  run();
});

function buildExampleButtons() {
  const box = $("#examples");
  window.EXAMPLES.forEach((ex, idx) => {
    const b = document.createElement("button");
    b.textContent = ex.name;
    b.addEventListener("click", () => { $("#sql").value = ex.sql; run(); });
    box.appendChild(b);
  });
}

function config() {
  return {
    threshold: Number($("#threshold").value || 10),
    algorithm: $("#algorithm").value,
    use_histograms: $("#use-hist").checked,
    use_multicolumn: $("#use-multicol").checked,
    allow_cartesian: $("#allow-cartesian").checked,
    run: $("#run-sqlite").checked,
  };
}

// ---------------------------------------------------------------------------
// Main request
// ---------------------------------------------------------------------------
async function run() {
  showMessage(null);
  const sql = $("#sql").value.trim();
  if (!sql) { showMessage("请输入 SQL", "error"); return; }
  const btn = $("#btn-run");
  btn.disabled = true; btn.textContent = "优化中…";
  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql, ...config() }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      showMessage(data.error || "请求失败", "error");
      if (data.trace) console.error(data.trace);
      return;
    }
    state.payload = data;
    renderAll(data);
  } catch (err) {
    showMessage(String(err), "error");
  } finally {
    btn.disabled = false; btn.textContent = "优化 (⌘/Ctrl+Enter)";
  }
}

async function runBruteForce() {
  const sql = $("#sql").value.trim();
  if (!sql) return;
  const box = $("#brute");
  box.innerHTML = '<span class="hint">穷举中…</span>';
  try {
    const res = await fetch("/api/brute-force", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql, ...config() }),
    });
    const data = await res.json();
    if (!res.ok || data.error) { box.innerHTML = `<span class="error">${data.error || "失败"}</span>`; return; }
    renderBruteForce(data);
  } catch (err) {
    box.innerHTML = `<span class="error">${err}</span>`;
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderAll(data) {
  renderMeta(data);
  renderGraph(data);
  renderPlan(data);
  renderEstimates(data);
  renderEnumeration(data);
  renderPredicates(data);
  renderNotes(data);
}

function showMessage(text, kind = "info") {
  const box = $("#message");
  if (!text) { box.className = "message hidden"; box.textContent = ""; return; }
  box.className = `message ${kind}`;
  box.textContent = text;
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined) return "—";
  if (Number.isInteger(n)) return n.toLocaleString("zh-CN");
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString("zh-CN");
  return Number(n).toFixed(digits);
}

function pct(x) {
  if (x === null || x === undefined) return "—";
  return (x * 100).toFixed(x < 0.01 ? 3 : 1) + "%";
}

function srcTag(source) {
  if (!source) return "";
  const cls = source === "multicol" ? "multicol" : source;
  const label = {
    histogram: "直方图", mcv: "高频值", ndv: "NDV",
    multicol: "多列统计", default: "默认值", overlap: "重叠",
  }[source] || source;
  return `<span class="src ${cls}">${label}</span>`;
}

function renderMeta(data) {
  const q = data.query;
  const chips = [];
  chips.push(`<span class="chip">形状 ${shapeName(q.shape)}</span>`);
  chips.push(`<span class="chip">${data.atom_count} 个连接原子</span>`);
  chips.push(`<span class="chip">算法 ${algoName(data.algorithm)}</span>`);
  chips.push(`<span class="chip">总代价 ${fmt(data.cost, 2)}</span>`);
  chips.push(`<span class="chip">估算行数 ${fmt(data.est_rows)}</span>`);
  (data.stages || []).forEach((s, i) => chips.push(`<span class="chip stage">${i + 1}. ${s}</span>`));
  const semi = (q.semi_units || []).map((u) =>
    `<span class="chip">${u.kind === "semi" ? "⋉" : "▷"} ${u.inner.join("+")} ← ${u.outer_refs.join(",")}</span>`);
  const meta = `
    <div class="chips">${chips.join("")}${semi.join("")}</div>
    <div>${q.tables.map((t) => `<code>${t.alias}</code> → ${t.table}`).join(" · ")}</div>
    <div>解析 + 优化耗时 <b>${fmt(data.parse_ms, 2)} ms</b> · 阈值 <b>${data.threshold}</b> ·
      ${data.exhaustive_available ? "本次使用穷举 DP（精确）" : "原子数超阈值，已退回贪心"}
    </div>`;
  $("#query-meta").innerHTML = meta;
}

function shapeName(shape) {
  return { chain: "链 chain", star: "星 star", cycle: "环 cycle",
           clique: "团 clique", mixed: "混合 mixed" }[shape] || shape;
}

function algoName(a) {
  return { exhaustive: "穷举 DP", "exhaustive(forced)": "穷举 DP（强制）",
           greedy: "贪心", brute: "暴力" }[a] || a;
}

// ---------------------------------------------------------------------------
// Join graph
// ---------------------------------------------------------------------------
function renderGraph(data) {
  const svg = $("#graph-svg");
  const nodes = data.join_graph.nodes;
  const edges = data.join_graph.edges;
  const W = 900, H = 420;
  svg.innerHTML = "";
  if (!nodes.length) { $("#graph-info").textContent = ""; return; }

  // circular layout, deterministic
  const cx = W / 2, cy = H / 2;
  const radius = Math.min(W, H) / 2 - 70;
  const positions = {};
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    positions[n.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });

  const NS = "http://www.w3.org/2000/svg";
  const order = new Set((data.order || []));
  // edges first
  edges.forEach((e) => {
    if (!positions[e.source] || !positions[e.target]) return;
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", `edge ${e.kind}`);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", positions[e.source].x);
    line.setAttribute("y1", positions[e.source].y);
    line.setAttribute("x2", positions[e.target].x);
    line.setAttribute("y2", positions[e.target].y);
    const title = document.createElementNS(NS, "title");
    title.textContent = `${e.predicates.join(" AND ")}\n选择率 ${pct(e.selectivity)} · 来源 ${e.source_of_estimate || "—"}`;
    g.appendChild(line); g.appendChild(title);
    const mx = (positions[e.source].x + positions[e.target].x) / 2;
    const my = (positions[e.source].y + positions[e.target].y) / 2;
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", mx); t.setAttribute("y", my - 4);
    t.setAttribute("text-anchor", "middle");
    t.textContent = pct(e.selectivity);
    g.appendChild(t);
    svg.appendChild(g);
  });
  // nodes
  nodes.forEach((n) => {
    const p = positions[n.id];
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node" + (order.has(n.id) ? " selected" : ""));
    const rect = document.createElementNS(NS, "rect");
    const w = 120, h = 46;
    rect.setAttribute("x", p.x - w / 2); rect.setAttribute("y", p.y - h / 2);
    rect.setAttribute("width", w); rect.setAttribute("height", h);
    rect.setAttribute("rx", 9);
    const name = document.createElementNS(NS, "text");
    name.setAttribute("class", "name");
    name.setAttribute("x", p.x); name.setAttribute("y", p.y - 3);
    name.setAttribute("text-anchor", "middle");
    name.textContent = `${n.id} · ${n.table}`;
    const rows = document.createElementNS(NS, "text");
    rows.setAttribute("class", "rows");
    rows.setAttribute("x", p.x); rows.setAttribute("y", p.y + 15);
    rows.setAttribute("text-anchor", "middle");
    rows.textContent = `${fmt(n.rows)} / ${fmt(n.base_rows)} 行`;
    const title = document.createElementNS(NS, "title");
    title.textContent = (n.filters || []).map((f) => `${f.text}  ${pct(f.selectivity)} (${f.source})`).join("\n") || "无过滤条件";
    g.appendChild(rect); g.appendChild(name); g.appendChild(rows); g.appendChild(title);
    svg.appendChild(g);
  });
  $("#graph-info").textContent =
    `节点为表别名，第二行是过滤后 / 基础行数；边上数字是连接选择率。选中（绿色）节点按最优顺序排列。`;
}

// ---------------------------------------------------------------------------
// Plan tree
// ---------------------------------------------------------------------------
function renderPlan(data) {
  const plan = data.plan;
  $("#plan-summary").innerHTML =
    `总代价 <b>${fmt(data.cost, 2)}</b> · 输出行数 <b>${fmt(data.est_rows)}</b> · 连接顺序 <b>${(data.order || []).join(" → ")}</b>`;
  const tree = $("#plan-tree");
  tree.innerHTML = "";
  tree.appendChild(planNode(plan));
}

function planNode(node) {
  const ul = document.createElement("ul");
  const li = document.createElement("li");
  const algorithms = node.algorithm ? `<span class="badge ${node.algorithm}">${node.algorithm}</span>` : "";
  const preds = (node.predicates || [])
    .map((p) => `<span title="${escapeHtml(p.note || "")}">${escapeHtml(p.text)} ${srcTag(p.source)}</span>`)
    .join(" ");
  const filters = node.type === "scan" && node.filters && node.filters.length
    ? node.filters.map((f) => `<span title="${escapeHtml(f.note || "")}">${escapeHtml(f.text)} ${srcTag(f.source)}</span>`).join(" ")
    : "";
  const cand = Object.keys(node.candidates || {}).length
    ? `<span class="cost" title="候选算法代价">${Object.entries(node.candidates)
        .map(([k, v]) => `${k}:${fmt(v, 1)}`).join(" / ")}</span>`
    : "";
  const semi = node.semi_selectivity !== undefined
    ? `<span class="src ${node.semi_source || "default"}">保留 ${pct(node.semi_selectivity)}</span>` : "";
  li.innerHTML = `
    <div class="node">
      <span class="badge">${node.type}</span>
      <b>${escapeHtml(node.label)}</b>
      ${algorithms}
      <span class="rows">rows≈${fmt(node.est_rows)}</span>
      <span class="cost">cost ${fmt(node.cost, 2)}</span>
      ${cand}${semi}
      ${preds}${filters}
    </div>`;
  (node.children || []).forEach((child) => li.appendChild(planNode(child)));
  ul.appendChild(li);
  return ul;
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------------------------------------------------------------------------
// Estimates vs actual
// ---------------------------------------------------------------------------
function renderEstimates(data) {
  const box = $("#estimate-table");
  const v = data.validation || {};
  if (!v.okay) {
    box.innerHTML = `<div class="message error">未在 SQLite 上执行：${escapeHtml(v.error || "未启用")}</div>`;
  } else {
    const rows = (v.steps || []).map((s) => {
      const ratio = s.ratio;
      const cls = ratio === null || ratio === undefined ? "" : (ratio > 1.5 || ratio < 0.67 ? "bad" : (ratio > 1.2 || ratio < 0.83 ? "warn" : ""));
      const barW = ratio === null || ratio === undefined ? 0 : Math.min(100, Math.max(2, (Math.min(ratio, 4) / 4) * 100));
      return `<tr>
        <td>${escapeHtml(s.node)}</td>
        <td>${s.relations.join(", ")}</td>
        <td class="num">${fmt(s.est_rows)}</td>
        <td class="num">${fmt(s.actual_rows)}</td>
        <td class="num ${ratio > 1.2 ? "neg" : (ratio < 0.83 ? "neg" : "pos")}">
          ${ratio === null || ratio === undefined ? "—" : ratio.toFixed(2) + "×"}</td>
        <td><div class="bar ${cls}"><span style="width:${barW}%"></span></div></td>
        <td class="num">${fmt(s.error_pct, 1)}%</td>
      </tr>`;
    }).join("");
    const total = (v.steps || []).length;
    const worst = (v.steps || []).filter((s) => s.error_pct != null)
      .sort((a, b) => b.error_pct - a.error_pct)[0];
    box.innerHTML = `
      <table>
        <thead><tr>
          <th>节点</th><th>关系集合</th><th class="num">估算行数</th>
          <th class="num">实际行数</th><th class="num">偏差倍数</th>
          <th>偏差分布</th><th class="num">偏差%</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="meta">共 ${total} 个节点 · 最大偏差
        ${worst ? `${escapeHtml(worst.node)}，估算 ${fmt(worst.est_rows)} 实际 ${fmt(worst.actual_rows)}（${fmt(worst.error_pct, 1)}%）` : "无"}
      </div>`;
  }
  // timing boxes
  const t = data.timing || {};
  const boxes = [];
  if (t.chosen_ms != null) boxes.push(`<div class="tbox best"><div class="label">最优计划</div><div class="value">${t.chosen_ms.toFixed(2)} ms</div></div>`);
  if (t.slow_ms != null) boxes.push(`<div class="tbox"><div class="label">最差顺序执行（对照）</div><div class="value">${t.slow_ms.toFixed(2)} ms</div><div class="hint">${(t.slow_order || []).join(" → ")}</div></div>`);
  if (t.slow_ms == null && t.slow_order && t.slow_order.length) boxes.push(`<div class="tbox"><div class="label">最差顺序（超时 &gt;3s）</div><div class="value" style="font-size:12px">${t.slow_order.join(" → ")}</div></div>`);
  if (t.sqlite_ms != null) boxes.push(`<div class="tbox"><div class="label">SQLite 自行选择顺序</div><div class="value">${t.sqlite_ms.toFixed(2)} ms</div></div>`);
  if (t.slow_error) boxes.push(`<div class="tbox"><div class="label">对照不可用</div><div class="value" style="font-size:12px">${escapeHtml(t.slow_error)}</div></div>`);
  $("#timing").innerHTML = boxes.join("");
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------
function renderEnumeration(data) {
  const q = data.query;
  const bf = data.brute_force || {};
  $("#enum-summary").innerHTML =
    `本次查询：连接原子 <b>${data.atom_count}</b>，DP 子问题 <b>${data.subproblems}</b>，转移 <b>${data.transitions}</b>；` +
    `穷举所有顺序需要 <b>${factorial(data.atom_count)}</b> 种排列。`;

  const rows = [
    ["连接原子数", data.atom_count],
    ["DP 子问题（连通子集）", data.subproblems],
    ["DP 转移（子集扩展）", data.transitions],
    ["全部排列（暴力）", factorial(data.atom_count)],
    ["贪心阈值", data.threshold],
  ].map(([k, v]) => `<tr><td>${k}</td><td class="num">${fmt(v, 0)}</td></tr>`).join("");
  $("#enum-query").innerHTML = `<table><tbody>${rows}</tbody></table>`;

  // candidate plans
  const alts = data.alternatives || [];
  $("#alternatives").innerHTML = alts.length ? `
    <table><thead><tr><th>顺序</th><th class="num">代价</th><th class="num">行数</th></tr></thead>
    <tbody>${alts.map((a, i) => `<tr class="${i === 0 ? "best" : ""}">
      <td>${a.order.join(" → ")}</td>
      <td class="num">${fmt(a.cost, 2)}</td>
      <td class="num">${fmt(a.rows)}</td></tr>`).join("")}</tbody></table>`
    : '<div class="hint">本次没有记录到候选计划</div>';

  // exhaustive-vs-brute-force badge
  const badge = $("#brute");
  if (bf.skipped) {
    badge.innerHTML = `<div class="hint">${escapeHtml(bf.reason || "已跳过")}</div>`;
  } else {
    badge.innerHTML = `
      <div class="chips">
        <span class="chip">合法排列 ${bf.perms}</span>
        <span class="chip">暴力最优 ${fmt(bf.cost, 4)}</span>
        <span class="chip">DP 最优 ${fmt(bf.dp_cost, 4)}</span>
        <span class="chip best">${bf.match ? "✓ 与穷举一致" : "✗ 不一致"}</span>
      </div>
      <div class="meta">暴力顺序 ${bf.order.join(" → ")}<br>DP 顺序 ${bf.dp_order.join(" → ")}</div>`;
  }

  const g = data.greedy;
  const ex = data.exact_optimum;
  if (g && data.algorithm === "greedy") {
    $("#alternatives").innerHTML += `<div class="meta">
      采用贪心：${g.order.join(" → ")}，代价 ${fmt(g.cost, 2)}` +
      (ex ? `；精确最优 ${fmt(ex.cost, 2)}（${ex.order.join(" → ")}），差距 ${(ex.gap * 100).toFixed(2)}%` : "") +
      `</div>`;
  } else if (g) {
    $("#alternatives").innerHTML += `<div class="meta">贪心结果：${g.order.join(" → ")}，代价 ${fmt(g.cost, 2)}
      （相对最优 ${(g.cost / data.cost).toFixed(2)}×）</div>`;
  }
  renderEnumerationShapes();
}

function factorial(n) {
  if (n > 18) return "≈" + Number(10 ** (n * (Math.log10(n) - Math.log10(Math.E)) + 0.5 * Math.log10(2 * Math.PI * n))).toExponential(2);
  let r = 1; for (let i = 2; i <= n; i++) r *= i; return r;
}

async function loadEnumeration() {
  try {
    const res = await fetch("/api/enumeration");
    state.enumeration = await res.json();
    renderEnumerationShapes();
  } catch (e) { /* ignore */ }
}

function renderEnumerationShapes() {
  const data = state.enumeration;
  const table = $("#enum-shapes");
  if (!data) { table.innerHTML = ""; return; }
  const sizes = data.sizes;
  const head = `<tr><th>形状</th>${sizes.map((n) => `<th class="num">n=${n}</th>`).join("")}</tr>`;
  const rows = data.shapes.map((shape) => {
    const cells = sizes.map((n) => {
      const r = data.table[shape].find((x) => x.n === n);
      if (!r) return "<td class='num'>—</td>";
      return `<td class="num" title="排列 ${r.permutations} · 转移 ${r.dp_transitions} · 边 ${r.edges}">
        <b>${r.dp_states}</b><br><span class="hint">${r.dp_transitions}</span></td>`;
    }).join("");
    return `<tr><th class="shape">${shapeName(shape)}</th>${cells}</tr>`;
  }).join("");
  table.innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>
    <tfoot><tr><td colspan="${sizes.length + 1}" class="hint">
      每格：<b>粗体</b> = DP 子问题数（连通子集个数）；下方小字 = DP 转移数。
      链/环/团 的边数随 n 增长更快，子问题数也更多；星形最少。
    </td></tr></tfoot>`;
}

function renderBruteForce(data) {
  if (data.skipped) {
    $("#brute").innerHTML = `<div class="hint">${escapeHtml(data.reason)}<br>DP 代价 ${fmt(data.dp_cost, 2)}，顺序 ${data.dp_order.join(" → ")}</div>`;
    return;
  }
  const status = data.match
    ? `<span class="best">✓ 与穷举全体排列一致</span>`
    : `<span style="color:var(--warn)">✗ 不一致（差 ${fmt(Math.abs(data.brute_cost - data.dp_cost), 4)}）</span>`;
  $("#brute").innerHTML = `
    <div class="chips">
      <span class="chip">穷举排列 ${data.perms}</span>
      <span class="chip">暴力最优 ${fmt(data.brute_cost, 4)}</span>
      <span class="chip">DP 最优 ${fmt(data.dp_cost, 4)}</span>
    </div>
    <div class="meta">暴力顺序 ${data.brute_order.join(" → ")}<br>
      DP 顺序 ${data.dp_order.join(" → ")}<br>${status}</div>`;
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------
function renderPredicates(data) {
  const preds = data.predicates || [];
  if (!preds.length) { $("#predicates").innerHTML = '<div class="hint">没有谓词</div>'; return; }
  const scopeName = { join: "连接", "inner-join": "子查询内连接",
    correlation: "子查询关联", "subquery-filter": "子查询过滤",
    filter: "过滤", "outer-join": "外连接" };
  const rows = preds.map((p) => `<tr>
    <td><code>${escapeHtml(p.text)}</code></td>
    <td>${scopeName[p.scope] || p.scope}</td>
    <td class="num">${p.selectivity == null ? "—" : pct(p.selectivity)}</td>
    <td>${srcTag(p.source)}</td>
    <td class="hint">${escapeHtml(p.note || "")}</td>
  </tr>`).join("");
  $("#predicates").innerHTML = `
    <table><thead><tr><th>谓词</th><th>作用域</th><th class="num">选择率</th>
      <th>来源</th><th>说明</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// Notes / rules
// ---------------------------------------------------------------------------
function renderNotes(data) {
  const notes = data.notes || [];
  const rules = [
    "外连接两侧不能交换：LEFT JOIN 的右侧（nullable）永远在保留侧之后求值；RIGHT JOIN 归一化为 LEFT JOIN。",
    "半连接 / 反连接由关联子查询改写而来，可以作为原子参与换序，但必须在其外引用都可用之后应用。",
    "每个中间结果的行数 = Π 基础行数 × Π 边选择率，与连接顺序无关，因此子集 DP 与暴力枚举最优解一致。",
    "连接算法按代价择一：嵌套循环（小外表）/ 哈希连接（默认）/ 排序归并（输入已有序或输出需有序）。",
  ];
  const items = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")
    + rules.map((r) => `<li class="rule">${r}</li>`).join("");
  $("#notes").innerHTML = items;
}
