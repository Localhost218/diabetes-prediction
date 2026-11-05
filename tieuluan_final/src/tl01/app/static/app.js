// Giao diện Web App: mỗi phân hệ gọi một endpoint POST /api/<tên> và vẽ kết quả.
const $ = (s) => document.querySelector(s);
const fmt = (v, d = 2) => Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (p) => fmt(100 * p, 1) + " %";

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === b.dataset.tab));
}));

function probBar(label, p, color) {
  return `<div class="row"><span>${label}</span><b>${pct(p)}</b></div>
          <div class="bar"><i style="width:${(100 * p).toFixed(1)}%;background:${color}"></i></div>`;
}

// ------------------------------------------------------------------ biểu đồ đường SVG
function lineChart(el, series, opts = {}) {
  const W = 760, H = 330, L = 52, R = 14, T = 14, B = 34;
  const all = series.flatMap((s) => s.y);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const n = series[0].y.length;
  const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#2c3b5c"/>
          <text x="${L - 6}" y="${y(v) + 4}" fill="#8fa1c2" font-size="11" text-anchor="end">${fmt(v, opts.dec ?? 1)}</text>`;
  }
  (opts.xlabels || []).forEach((t, i, a) => {
    if (i % Math.ceil(a.length / 6) === 0) g += `<text x="${x(i)}" y="${H - 10}" fill="#8fa1c2" font-size="11" text-anchor="middle">${t}</text>`;
  });
  const paths = series.map((s) => {
    const d = s.y.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.w || 2}" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
  }).join("");
  const legend = series.map((s, i) => `<rect x="${L + 10 + i * 150}" y="${T + 4}" width="14" height="4" fill="${s.color}"/>
     <text x="${L + 30 + i * 150}" y="${T + 10}" fill="#e6ecf7" font-size="12">${s.name}</text>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}${paths}${legend}</svg>`;
}

// ------------------------------------------------------------------ Tiểu đường
$("#d-run").onclick = async () => {
  const r = await post("/api/diabetes", {
    gender: $("#d-gender").value, age: +$("#d-age").value, hypertension: +$("#d-hyp").value,
    heart_disease: +$("#d-heart").value, smoking_history: $("#d-smoke").value, bmi: +$("#d-bmi").value,
    HbA1c_level: +$("#d-hba").value, blood_glucose_level: +$("#d-glu").value });
  const col = r.level === "cao" ? "var(--red)" : r.level === "trung bình" ? "var(--amber)" : "var(--green)";
  $("#d-out").innerHTML = `<h3>Kết quả sàng lọc</h3>
    <div class="big" style="color:${col}">${pct(r.p_mlp)}</div>
    <span class="badge" style="background:${col};color:#0f1726">Nguy cơ ${r.level}</span>
    <div style="margin-top:14px">${probBar("Mạng MLP (PyTorch)", r.p_mlp, "var(--accent)")}${probBar("Random Forest", r.p_rf, "var(--violet)")}</div>
    <p>${r.advice}</p><p class="muted">Thời gian suy luận: ${fmt(r.ms)} ms · công cụ hỗ trợ, không thay chẩn đoán của bác sĩ.</p>`;
};

// ------------------------------------------------------------------ Giá nhà
$("#h-run").onclick = async () => {
  const r = await post("/api/house", {
    province: $("#h-prov").value, district: $("#h-dist").value, area: +$("#h-area").value, floors: +$("#h-floors").value,
    bedrooms: +$("#h-bed").value, bathrooms: +$("#h-bath").value, frontage: +$("#h-front").value, road: +$("#h-road").value,
    legal: $("#h-legal").value, furniture: $("#h-furn").value });
  $("#h-out").innerHTML = `<h3>Giá ước tính</h3>
    <div class="big">${fmt(r.p_rf)} <small style="font-size:18px">tỷ đồng</small></div>
    <p class="muted">Khoảng hợp lý (± RMSE kiểm thử): ${fmt(r.low)} – ${fmt(r.high)} tỷ</p>
    <div class="kpis"><div class="kpi"><b>${fmt(r.p_rf)}</b><span>Random Forest (tỷ)</span></div>
      <div class="kpi"><b>${fmt(r.p_mlp)}</b><span>MLP PyTorch (tỷ)</span></div>
      <div class="kpi"><b>${fmt(r.per_m2, 1)}</b><span>Triệu đồng / m²</span></div></div>
    <p class="muted">Thời gian suy luận: ${fmt(r.ms)} ms</p>`;
};

// ------------------------------------------------------------------ Cảm xúc
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("#s-text").value = c.dataset.t; }));
$("#s-run").onclick = async () => {
  const r = await post("/api/sentiment", { text: $("#s-text").value });
  const cols = ["var(--red)", "var(--muted)", "var(--green)"];
  const k = r.labels.indexOf(r.label);
  $("#s-out").innerHTML = `<h3>Sắc thái</h3>
    <div class="big" style="color:${cols[k]}">${r.label}</div>
    <p class="muted">Điểm hài lòng kỳ vọng: <b style="color:#fff">${fmt(r.score)} / 5</b></p>
    ${r.labels.map((l, i) => probBar(l + " (MLP)", r.p_mlp[i], cols[i])).join("")}
    <p class="muted">Logistic Regression: ${r.labels.map((l, i) => `${l} ${pct(r.p_lr[i])}`).join(" · ")}</p>
    <div class="words">Từ đóng góp: ${r.words.map((w) => `<span>${w[0]}</span>`).join("") || "—"}</div>
    <p class="muted">Thời gian suy luận: ${fmt(r.ms)} ms</p>`;
};

// ------------------------------------------------------------------ Thị giác
let uploaded = null;
$("#v-file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader(); rd.onload = () => { uploaded = rd.result; }; rd.readAsDataURL(f);
};
$("#v-rand").onclick = () => { $("#v-idx").value = Math.floor(Math.random() * 5000); uploaded = null; $("#v-run").click(); };
$("#v-run").onclick = async () => {
  const r = await post("/api/vision", { dataset: $("#v-ds").value, arch: $("#v-arch").value, index: +$("#v-idx").value, image: uploaded });
  const ok = r.truth === null ? "" : r.truth === r.top[0].label ? "var(--green)" : "var(--red)";
  $("#v-out").innerHTML = `<h3>Kết quả nhận diện</h3>
    <div class="imgs"><figure><img src="${r.image}"><figcaption>Ảnh đầu vào</figcaption></figure>
      <figure><img src="${r.cam}"><figcaption>Bản đồ CAM của lớp dự đoán</figcaption></figure></div>
    <p style="margin:12px 0 6px"><span class="big" style="font-size:26px">${r.top[0].label}</span>
      ${r.truth !== null ? `<span class="badge" style="background:${ok};color:#0f1726">Nhãn thật: ${r.truth}</span>` : ""}</p>
    ${r.top.map((t, i) => probBar(t.label, t.p, i ? "var(--violet)" : "var(--accent)")).join("")}
    <p class="muted">Độ chính xác kiểm thử của mô hình: ${pct(r.acc)} · suy luận ${fmt(r.ms)} ms trên CPU</p>`;
};

// ------------------------------------------------------------------ Cổ phiếu
let cell = "lstm";
document.querySelectorAll("#st-cell button").forEach((b) => b.addEventListener("click", () => {
  cell = b.dataset.c; document.querySelectorAll("#st-cell button").forEach((x) => x.classList.toggle("on", x === b));
}));
$("#st-days").oninput = (e) => { $("#st-dv").textContent = e.target.value; };
$("#st-run").onclick = async () => {
  const r = await post("/api/stock", { cell, days: +$("#st-days").value });
  lineChart($("#st-chart"), [{ name: "Giá thực tế", y: r.true, color: "#e6ecf7", w: 1.6 },
    { name: `Dự báo ${r.label}`, y: r.pred, color: "#3b82f6", w: 2, dash: true }], { xlabels: r.dates.map((d) => d.slice(2)) });
  const up = r.next_ret >= 0;
  $("#st-out").innerHTML = `<h3>Phiên kế tiếp sau ${r.last_date}</h3>
    <div class="big" style="color:${up ? "var(--green)" : "var(--red)"}">${fmt(r.next_price)} <small style="font-size:16px">nghìn đ</small></div>
    <p class="muted">Đóng cửa gần nhất ${fmt(r.last_close)} · thay đổi dự báo ${up ? "+" : ""}${fmt(r.next_ret, 2)} %</p>
    <div class="kpis"><div class="kpi"><b>${fmt(r.rmse, 3)}</b><span>RMSE kiểm thử</span></div>
      <div class="kpi"><b>${fmt(r.naive_rmse, 3)}</b><span>RMSE “giữ nguyên giá”</span></div>
      <div class="kpi"><b>${pct(r.dir_acc)}</b><span>Đúng chiều tăng/giảm</span></div></div>
    <div class="kpis"><div class="kpi"><b>${fmt(r.rsi, 1)}</b><span>RSI-14 hiện tại</span></div>
      <div class="kpi"><b>${fmt(r.vol30, 2)} %</b><span>Biến động 30 phiên</span></div>
      <div class="kpi"><b>${fmt(r.ms)} ms</b><span>Suy luận ${r.dates.length} cửa sổ</span></div></div>
    <p class="muted">Minh họa học thuật, không phải khuyến nghị đầu tư.</p>`;
};

// ------------------------------------------------------------------ Rời bỏ
let group = "any", seed = 3;
document.querySelectorAll("#c-group button").forEach((b) => b.addEventListener("click", () => {
  group = b.dataset.g; document.querySelectorAll("#c-group button").forEach((x) => x.classList.toggle("on", x === b));
}));
$("#c-run").onclick = async () => {
  const r = await post("/api/churn", { group, seed: seed++ });
  lineChart($("#c-chart"), [{ name: "Phút nghe mỗi ngày", y: r.secs.map((s) => s / 60), color: "#10b981" },
    { name: "Số bài khác nhau", y: r.unq, color: "#f59e0b", w: 1.5 }], { xlabels: r.secs.map((_, i) => "3/" + (i + 1)), dec: 0 });
  const cells = Object.keys(r.probs);
  const mean = cells.reduce((s, c) => s + r.probs[c], 0) / cells.length;
  const votes = cells.filter((c) => r.probs[c] >= r.thr[c]).length;
  $("#c-out").innerHTML = `<h3>Người dùng #${r.index} trong tập kiểm thử</h3>
    <div class="big" style="color:${votes >= 2 ? "var(--red)" : "var(--green)"}">${votes >= 2 ? "Nguy cơ rời bỏ" : "Khả năng ở lại"}</div>
    <p class="muted">Nhãn thật: <b style="color:#fff">${r.truth ? "đã rời bỏ" : "ở lại"}</b> · ${votes}/4 mô hình vượt ngưỡng · ${r.active.filter((a) => a > 0).length}/31 ngày hoạt động</p>
    ${cells.map((c) => probBar(`${r.labels[c]} (ngưỡng ${fmt(r.thr[c])})`, r.probs[c], r.probs[c] >= r.thr[c] ? "var(--red)" : "var(--accent)")).join("")}
    <p class="muted">Xác suất trung bình ${pct(mean)} · suy luận 4 mô hình ${fmt(r.ms)} ms</p>`;
};
