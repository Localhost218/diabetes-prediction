// Giao diện Web App Tiểu luận 02: không dùng thư viện ngoài, đồ thị và biểu đồ vẽ bằng SVG thuần.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body) }).then((r) => r.json());

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === t.dataset.tab));
  if (t.dataset.tab === "kg") drawKG($("#kg-name").value);
}));
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
  const pane = c.closest(".pane").id;
  $(pane === "health" ? "#h-q" : "#f-q").value = c.dataset.q;
}));

function renderAgent(prefix, r) {
  $(`#${prefix}-trace`).innerHTML = r.trace.map((s) => {
    if (s.type === "ACTION") {
      return `<li class="ACTION"><b>ACTION</b><span class="kind ${s.kind}">${s.kind}</span> <code>${s.tool}</code>
        <span class="muted">(${s.ms.toFixed(0)} ms)</span></li>`;
    }
    return `<li class="${s.type}"><b>${s.type}</b> ${esc(s.text || "")}</li>`;
  }).join("");
  $(`#${prefix}-ans`).textContent = r.answer;
  const v = r.verification;
  $(`#${prefix}-ver`).textContent = v.used === "llm"
    ? `Qwen2.5 · ${v.n_citations} trích dẫn · trung thực số ${(100 * v.numeric_faithfulness).toFixed(0)}% · ${(r.total_ms / 1000).toFixed(1)} s`
    : `bản tổng hợp theo khuôn · ${(r.total_ms / 1000).toFixed(1)} s`;
  $(`#${prefix}-ev`).innerHTML = r.evidence.map(([src, t]) => `<li><code>${esc(src)}</code> ${esc(t)}</li>`).join("");
  return r;
}

$("#h-run").addEventListener("click", async () => {
  $("#h-trace").innerHTML = "<li class='muted'>Agent đang chạy chu trình ReAct...</li>";
  const body = { question: $("#h-q").value };
  if ($("#h-useprof").value === "1") {
    Object.assign(body, { HbA1c_level: +$("#h-hba1c").value, blood_glucose_level: +$("#h-glu").value,
      bmi: +$("#h-bmi").value, age: +$("#h-age").value, gender: $("#h-gender").value });
  }
  const r = renderAgent("h", await post("/api/agent", body));
  const risk = r.observations.chan_doan_nguy_co;
  $("#h-risk").innerHTML = risk ? `<div class="gauge"><div><span class="muted">P(đái tháo đường)</span><br>
    <b>${(100 * risk.probability).toFixed(1)}%</b></div><div class="meter"><i style="left:${(100 * risk.probability).toFixed(1)}%"></i></div>
    <div><span class="muted">Nguy cơ</span><br><b style="font-size:18px">${risk.risk_level}</b></div></div>` : "";
});

$("#f-run").addEventListener("click", async () => {
  $("#f-trace").innerHTML = "<li class='muted'>Agent đang chạy chu trình ReAct...</li>";
  renderAgent("f", await post("/api/agent", { question: $("#f-q").value }));
});

async function drawSeries(sym) {
  const d = await fetch(`/api/series/${sym}`).then((r) => r.json());
  const W = 520, H = 170, lo = Math.min(...d.v), hi = Math.max(...d.v);
  const x = (i) => 40 + (i / (d.v.length - 1)) * (W - 50), y = (v) => 10 + (1 - (v - lo) / (hi - lo)) * (H - 35);
  const pts = d.v.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const fmt = (v) => (sym === "GOLD_VND" ? v.toFixed(1) : (v / 1000).toFixed(1) + "k");
  $("#f-chart").innerHTML = `<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2563eb" stop-opacity=".35"/>
    <stop offset="1" stop-color="#2563eb" stop-opacity="0"/></linearGradient></defs>
    <polygon points="40,${H - 25} ${pts} ${W - 10},${H - 25}" fill="url(#g)"/>
    <polyline points="${pts}" fill="none" stroke="#2563eb" stroke-width="2"/>
    <text x="2" y="16" font-size="11" fill="#5d6b85">${fmt(hi)}</text><text x="2" y="${H - 26}" font-size="11" fill="#5d6b85">${fmt(lo)}</text>
    <text x="40" y="${H - 6}" font-size="11" fill="#5d6b85">${d.date[0]}</text>
    <text x="${W - 80}" y="${H - 6}" font-size="11" fill="#5d6b85">${d.date[d.date.length - 1]}</text>`;
}
$("#f-sym").addEventListener("change", (e) => drawSeries(e.target.value));
drawSeries("FPT");

const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0d9488", "#db2777", "#4f46e5", "#a16207",
  "#0891b2", "#65a30d", "#9333ea", "#ea580c", "#475569"];
async function drawKG(name) {
  const g = await fetch(`/api/kg/${name}`).then((r) => r.json());
  const types = [...new Set(g.nodes.map((n) => n.type))];
  const color = (t) => PALETTE[types.indexOf(t) % PALETTE.length];
  const pos = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
  $("#kg-stat").textContent = `${g.stats.nodes} thực thể · ${g.stats.edges} quan hệ · ${g.stats.predicates} vị từ`;
  const edges = g.edges.map((e) => `<line x1="${pos[e.s].x}" y1="${pos[e.s].y}" x2="${pos[e.o].x}" y2="${pos[e.o].y}"
    stroke="${e.p.includes("NGUOC") || e.p.includes("TUONG_TAC") || e.p.includes("HAN_CHE") ? "#f87171" : "#94a3b8"}"
    stroke-width="1.6" marker-end="url(#arr)"/>`).join("");
  const nodes = g.nodes.map((n) => `<g class="nd" data-id="${n.id}" style="cursor:pointer">
    <circle cx="${n.x}" cy="${n.y}" r="${7 + 2 * Math.sqrt(n.deg)}" fill="${color(n.type)}" stroke="#fff" stroke-width="2"/>
    <text x="${n.x}" y="${n.y - 11 - 2 * Math.sqrt(n.deg)}" text-anchor="middle" font-size="11" fill="#16233b">${esc(n.label.slice(0, 26))}</text></g>`).join("");
  $("#kg-svg").innerHTML = `<defs><marker id="arr" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>${edges}${nodes}`;
  $("#kg-legend").innerHTML = types.map((t) => `<span><i style="background:${color(t)}"></i>${t}</span>`).join("");
  const show = (id) => {
    const facts = g.edges.filter((e) => e.s === id || e.o === id);
    $("#kg-side").innerHTML = `<b>${esc(pos[id].label)}</b> <span class="muted">(${pos[id].type})</span><ul>` +
      facts.map((e) => `<li>${esc(e.text)} <span class="muted">· ${e.source || ""}</span></li>`).join("") + "</ul>";
  };
  document.querySelectorAll(".nd").forEach((el) => el.addEventListener("click", () => show(el.dataset.id)));
  show(name === "health" ? "Metformin" : "USD_VND");
}
$("#kg-name").addEventListener("change", (e) => drawKG(e.target.value));

$("#r-run").addEventListener("click", async () => {
  const r = await post("/api/rag", { q: $("#r-q").value });
  $("#r-out").innerHTML = Object.entries(r).map(([name, v]) => `<div class="col"><h4>${name}
    <span class="muted" style="font-size:12px;font-weight:400">${v.ms.toFixed(1)} ms</span></h4>` +
    v.hits.map((h, i) => `<div class="hit"><span class="score">${h.score.toFixed(3)}</span><span class="h">#${i + 1} ${esc(h.doc)}</span>
      <div>${esc(h.text)}</div></div>`).join("") + "</div>").join("");
});
