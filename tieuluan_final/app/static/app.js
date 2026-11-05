// Trang thực nghiệm: dựng form từ SPECS, gọi /api/<exp>/predict, hiển thị kết quả theo `kind`.
(() => {
  const { id: EXP_ID, spec: SPEC, color: COLOR } = window.EXP;
  const $ = (s, r = document) => r.querySelector(s);
  const byId = (id) => document.getElementById(id);   // tên trường có thể chứa dấu cách ("Locality Rating")
  const el = (tag, attrs = {}, html = "") => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v);
    }
    if (html) e.innerHTML = html;
    return e;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d) => {
    if (typeof v !== "number" || !isFinite(v)) return esc(v);
    const abs = Math.abs(v);
    const dd = d ?? (abs >= 1000 ? 0 : abs >= 10 ? 2 : abs >= 1 ? 3 : 4);
    return v.toLocaleString("vi-VN", { maximumFractionDigits: dd, minimumFractionDigits: Number.isInteger(v) ? 0 : Math.min(dd, 2) });
  };
  // nhãn tiếng Việt cho khoá kỹ thuật trong `details`; serve có thể ghi đè bằng SPECS[exp]["labels"]
  const LABELS = {
    architecture: "Kiến trúc", framework: "Framework", change_pct: "Thay đổi so với TB 7 phiên cuối (%)", mean_7: "Trung bình 7 phiên cuối",
    last_value: "Giá trị quan sát cuối", naive: "Dự báo ngây thơ (giữ nguyên)", naive_test: "Baseline ngây thơ trên tập test",
    test: "Mô hình trên tập test", target: "Biến mục tiêu", window: "Độ dài cửa sổ (bước)", unit: "Đơn vị",
    probability: "Xác suất lớp dương", threshold: "Ngưỡng quyết định", n_words: "Số từ", degree: "Bậc của nút",
    true_label: "Nhãn thật", node_id: "Mã nút", calls: "Số lượt gọi LLM", tokens: "Số token", steps: "Số bước",
    tokens_in: "Số token đầu vào", tokens_out: "Số token sinh ra",
    dir_acc: "đúng hướng", mae: "MAE", rmse: "RMSE", mape: "MAPE (%)", r2: "R²", accuracy: "Accuracy", f1: "F1",
    auc: "ROC-AUC", latency_ms: "Thời gian (ms)", model: "Mô hình", confidence: "Độ tin cậy",
    dataset: "Bộ dữ liệu", model_name: "Tên mô hình", input_size: "Kích thước đầu vào", model_input: "Ảnh mô hình nhìn thấy",
    correct: "Dự đoán đúng nhãn thật", split: "Thuộc tập", model_test_acc: "Accuracy mô hình trên tập test",
    ...(SPEC.labels || {}),
  };
  const lab = (k) => esc(LABELS[k] ?? k);
  const pct = (p) => (100 * p).toLocaleString("vi-VN", { maximumFractionDigits: 1, minimumFractionDigits: 1 }) + "%";
  const PCT_KEYS = new Set(["confidence", "model_test_acc"]);       // tỉ lệ 0..1 -> hiển thị phần trăm
  const ID_KEYS = new Set(["node_id"]);                             // mã định danh: không chèn dấu phân cách nghìn
  const modelName = (m) => (SPEC.model_names || {})[m] ?? m;

  // ------------------------------------------------------------ form
  const form = $("#form");
  const imageState = {};          // tên input -> Blob ảnh
  const models = SPEC.models || [];

  function field(inp) {
    const wrap = el("div", { class: "field" });
    const hint = [inp.min !== undefined ? `từ ${inp.min}` : "", inp.max !== undefined ? `đến ${inp.max}` : ""].filter(Boolean).join(" ");
    wrap.append(el("label", { for: "f-" + inp.name }, `${esc(inp.label || inp.name)} ${hint ? `<small>(${hint})</small>` : ""}`));
    const type = inp.type || "text";
    if (type === "select") {
      const s = el("select", { id: "f-" + inp.name, name: inp.name });
      for (const o of inp.options || []) {
        const [val, lab] = Array.isArray(o) ? o : typeof o === "object" ? [o.value, o.label] : [o, o];
        s.append(el("option", { value: val }, esc(lab)));
      }
      if (inp.default !== undefined) s.value = inp.default;
      wrap.append(s);
    } else if (type === "number") {
      wrap.append(el("input", { id: "f-" + inp.name, name: inp.name, type: "number", min: inp.min, max: inp.max,
        step: inp.step ?? "any", value: inp.default ?? "" }));
    } else if (type === "image") {
      wrap.append(imageField(inp));
    } else {
      const long = inp.multiline || SPEC.kind === "chat" || SPEC.kind === "text";
      const t = long ? el("textarea", { id: "f-" + inp.name, name: inp.name, placeholder: inp.placeholder || "" })
                     : el("input", { id: "f-" + inp.name, name: inp.name, type: "text", value: inp.default ?? "" });
      if (long && inp.default) t.value = inp.default;
      wrap.append(t);
    }
    return wrap;
  }

  function imageField(inp) {
    const box = el("div", { class: "imgbox" });
    const useCanvas = inp.canvas || /mnist/i.test(EXP_ID);
    if (useCanvas) {
      const cv = el("canvas", { width: 280, height: 280, id: "cv-" + inp.name });
      const ctx = cv.getContext("2d");
      const clear = () => { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280); delete imageState[inp.name]; };
      clear();
      let drawing = false;
      const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * 280 / r.width, (e.clientY - r.top) * 280 / r.height]; };
      cv.addEventListener("pointerdown", (e) => { drawing = true; const [x, y] = pos(e); ctx.beginPath(); ctx.moveTo(x, y); });
      cv.addEventListener("pointermove", (e) => {
        if (!drawing) return;
        const [x, y] = pos(e);
        Object.assign(ctx, { lineWidth: 20, lineCap: "round", lineJoin: "round", strokeStyle: "#fff" });
        ctx.lineTo(x, y); ctx.stroke();
      });
      const done = () => { if (drawing) { drawing = false; cv.toBlob((b) => (imageState[inp.name] = b)); } };
      cv.addEventListener("pointerup", done); cv.addEventListener("pointerleave", done);
      window.drawOnCanvas = (strokes) => {   // dùng bởi screenshots.py: vẽ nét chữ số theo toạ độ 0..1
        clear();
        Object.assign(ctx, { lineWidth: 22, lineCap: "round", lineJoin: "round", strokeStyle: "#fff" });
        for (const st of strokes) { ctx.beginPath(); st.forEach(([x, y], i) => (i ? ctx.lineTo(x * 280, y * 280) : ctx.moveTo(x * 280, y * 280))); ctx.stroke(); }
        return new Promise((res) => cv.toBlob((b) => { imageState[inp.name] = b; res(true); }));
      };
      box.append(el("div", {}, "<small>Vẽ chữ số bằng chuột/ngón tay, hoặc tải ảnh lên.</small>"), cv);
      const row = el("div", { class: "imgrow" });
      const bClear = el("button", { type: "button", class: "btn ghost small" }, "Xoá");
      bClear.onclick = clear;
      row.append(bClear);
      box.append(row);
    }
    const file = el("input", { type: "file", accept: "image/*", id: "f-" + inp.name, name: inp.name });
    const prev = el("img", { class: "preview", alt: "", style: "display:none" });
    file.onchange = () => {
      const f = file.files[0];
      if (!f) return;
      imageState[inp.name] = f;
      prev.src = URL.createObjectURL(f); prev.style.display = "";
      if (useCanvas) { const cv = byId("cv-" + inp.name); const img = new Image(); img.onload = () => cv.getContext("2d").drawImage(img, 0, 0, 280, 280); img.src = prev.src; }
    };
    box.append(file, prev);
    box.setImage = (blob) => { imageState[inp.name] = blob; prev.src = URL.createObjectURL(blob); prev.style.display = useCanvas ? "none" : "";
      if (useCanvas) { const cv = byId("cv-" + inp.name); const img = new Image(); img.onload = () => cv.getContext("2d").drawImage(img, 0, 0, 280, 280); img.src = prev.src; } };
    box.dataset.name = inp.name;
    return box;
  }

  // nhiều trường (giá nhà: 15) -> form hai cột để trang không quá dài
  if ((SPEC.inputs || []).length > 8) $(".panes").classList.add("wide");
  for (const inp of SPEC.inputs || []) form.append(field(inp));
  // serve đã có trường chọn mô hình riêng (vd. select "model") -> không vẽ thêm nhóm chip trùng
  const hasModelInput = (SPEC.inputs || []).some((i) => i.name === "model" || i.name === "model_id");
  if (models.length && !hasModelInput) {
    const wrap = el("div", { class: "field" });
    wrap.append(el("label", {}, "Mô hình"));
    const box = el("div", { class: "models" });
    models.forEach((m, i) => {
      const [val, lab] = typeof m === "object" ? [m.id ?? m.value, m.label ?? m.id] : [m, modelName(m)];
      box.append(el("label", {}, `<input type="radio" name="model_id" value="${esc(val)}" ${i === 0 ? "checked" : ""}><span>${esc(lab)}</span>`));
    });
    wrap.append(box);
    form.append(wrap);
  }

  async function fillExample() {
    const ex = SPEC.example || {};
    for (const inp of SPEC.inputs || []) {
      if (!(inp.name in ex)) continue;
      if (inp.type === "image") {
        const r = await fetch(`/api/${EXP_ID}/example/${encodeURIComponent(inp.name)}`);
        if (r.ok) form.querySelector(`.imgbox[data-name="${inp.name}"]`).setImage(await r.blob());
        continue;
      }
      const f = byId("f-" + inp.name);
      if (f) f.value = ex[inp.name];
    }
    if (ex.model_id) { const r = form.querySelector(`input[name=model_id][value="${ex.model_id}"]`); if (r) r.checked = true; }
  }
  $("#fill-example").onclick = fillExample;
  window.fillExample = fillExample;

  // ------------------------------------------------------------ gọi API
  const result = $("#result");
  async function run() {
    const fd = new FormData();
    for (const inp of SPEC.inputs || []) {
      if (inp.type === "image") { if (imageState[inp.name]) fd.append(inp.name, imageState[inp.name], "input.png"); continue; }
      const f = byId("f-" + inp.name);
      if (f && f.value !== "") fd.append(inp.name, f.value);
    }
    const m = form.querySelector("input[name=model_id]:checked");
    if (m) fd.append("model_id", m.value);
    const btn = $("#run");
    btn.disabled = true;
    result.className = "result";
    result.innerHTML = `<div class="loading"><div class="spin"></div>Mô hình đang suy luận…</div>`;
    try {
      const r = await fetch(`/api/${EXP_ID}/predict`, { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || r.statusText);
      render(data, fd);
      result.dataset.done = "1";
    } catch (e) {
      result.innerHTML = `<div class="errbox"><b>Lỗi:</b> ${esc(e.message)}</div>`;
      result.dataset.done = "err";
    } finally { btn.disabled = false; }
  }
  $("#run").onclick = run;

  // ------------------------------------------------------------ hiển thị
  const VISUAL_KEYS = new Set(["history", "forecast", "dates", "forecast_dates", "neighbors", "trace", "sources", "tools",
                               "proba", "topk", "image", "input_image", "answer", "x", "y", "actual", "pred", "unit"]);

  function probaEntries(res) {
    let p = res.proba || res.topk || res.probabilities;
    if (!p) return [];
    if (Array.isArray(p)) p = p.map((x) => (Array.isArray(x) ? x : [x.label ?? x.name, x.prob ?? x.p ?? x.score]));
    else p = Object.entries(p);
    return p.filter(([, v]) => typeof v === "number").sort((a, b) => b[1] - a[1]).slice(0, 10);
  }

  function bars(entries) {
    if (!entries.length) return "";
    return `<div class="bars">${entries.map(([k, v], i) => `
      <div class="bar ${i === 0 ? "top" : ""}"><span class="name" title="${esc(k)}">${esc(k)}</span>
        <div class="track"><div class="fill" style="width:${Math.max(1, 100 * v).toFixed(1)}%"></div></div>
        <span class="pct">${pct(v)}</span></div>`).join("")}</div>`;
  }

  function scalarRows(obj, skip = VISUAL_KEYS) {
    const rows = [];
    for (const [k, v] of Object.entries(obj || {})) {
      if (skip.has(k) || v === null || v === undefined) continue;
      if (typeof v === "object") {
        if (Array.isArray(v) && v.length <= 8 && v.every((x) => typeof x !== "object")) rows.push([k, v.map((x) => fmt(x)).join(", ")]);
        else if (!Array.isArray(v) && Object.keys(v).length <= 8 && Object.values(v).every((x) => typeof x !== "object"))
          rows.push([k, Object.entries(v).map(([a, b]) => `${lab(a)}: ${fmt(b)}`).join(" · ")]);
        continue;
      }
      if (typeof v === "string" && v.startsWith("data:image")) continue;
      if (typeof v === "boolean") { rows.push([k, v ? "✓ Đúng" : "✗ Sai"]); continue; }
      if (ID_KEYS.has(k)) { rows.push([k, esc(v)]); continue; }
      rows.push([k, PCT_KEYS.has(k) && typeof v === "number" ? pct(v) : fmt(v)]);
    }
    return rows.length ? `<table class="kv">${rows.map(([k, v]) => `<tr><td>${lab(k)}</td><td>${v}</td></tr>`).join("")}</table>` : "";
  }

  function metaLine(res) {
    const items = [];
    if (res.model) items.push(`Mô hình: <b>${esc(modelName(res.model))}</b>`);
    if (typeof res.latency_ms === "number") items.push(`Thời gian: <b>${fmt(res.latency_ms, 1)} ms</b>`);
    if (res.details && res.details.calls !== undefined) items.push(`Lượt gọi LLM: <b>${fmt(res.details.calls)}</b>`);
    return `<div class="meta">${items.map((s) => `<span>${s}</span>`).join("")}</div>`;
  }

  function verdict(res, subtitle) {
    // chuỗi thời gian: hiển thị giá trị số theo định dạng Việt thay cho nhãn chuỗi do serve định dạng sẵn
    let main = SPEC.kind === "series" && typeof res.value === "number" ? undefined : res.label ?? res.prediction ?? res.class;
    if (main === undefined && res.value !== undefined) {
      main = Array.isArray(res.value) ? fmt(res.value[0]) : fmt(res.value);
      const unit = res.unit || (res.details && res.details.unit) || SPEC.unit;
      if (unit) main += ` <small>${esc(unit)}</small>`;
    } else main = esc(main);
    const conf = probaEntries(res)[0];
    const sub = subtitle || (conf ? `Độ tin cậy ${pct(conf[1])}` : res.value !== undefined ? "Giá trị dự đoán" : "");
    return `<div class="verdict"><div><div class="sub">Kết quả</div><div class="big">${main}</div><div class="sub">${sub}</div></div></div>`;
  }

  function render(res, fd) {
    const kind = SPEC.kind;
    const d = res.details || {};
    let html = "";
    if (kind === "chat") html = renderChat(res, fd);
    else if (kind === "series") html = verdict(res, seriesSub(res)) + `<div class="chartbox"><canvas id="chart"></canvas></div>` + scalarRows({ ...d, ...pick(res) });
    else if (kind === "image") {
      const img = res.image || res.input_image || d.image;
      const blob = [...fd.values()].find((v) => v instanceof Blob);
      const src = img || (blob ? URL.createObjectURL(blob) : null);
      // ảnh đầu vào + mọi ảnh data-URL trong details (vd. model_input: ảnh sau tiền xử lý)
      const figs = src ? [[src, "Ảnh đầu vào"]] : [];
      for (const [k, v] of Object.entries(d)) if (typeof v === "string" && v.startsWith("data:image")) figs.push([v, LABELS[k] ?? k]);
      const figHtml = figs.map(([s, c]) => `<figure><img src="${s}" alt="${esc(c)}"><figcaption>${esc(c)}</figcaption></figure>`).join("");
      html = `<div class="split"><div class="figs">${figHtml}</div><div>${verdict(res)}${bars(probaEntries(res))}</div></div>` + scalarRows(d);
    } else if (kind === "graph") html = verdict(res) + bars(probaEntries(res)) + renderNeighbors(res, fd) + scalarRows(d);
    else html = verdict(res) + bars(probaEntries(res)) + scalarRows({ ...d, ...pick(res) });
    result.innerHTML = html + metaLine(res);
    if (kind === "series") drawChart(res);
  }

  function pick(res) {   // trường cấp ngoài không phải khoá chuẩn -> hiện trong bảng chi tiết
    const std = new Set(["label", "value", "proba", "details", "model", "latency_ms", "unit", "answer", "sources", "tools", "trace"]);
    return Object.fromEntries(Object.entries(res).filter(([k]) => !std.has(k)));
  }

  // ---- chuỗi thời gian
  function seriesData(res) {
    const d = res.details || {};
    const hist = d.history || res.history || d.actual || [];
    let fc = d.forecast || res.forecast || (Array.isArray(res.value) ? res.value : res.value !== undefined ? [res.value] : []);
    const hv = hist.map((x) => (typeof x === "object" ? x.value ?? x.y : x));
    const hd = d.dates || res.dates || hist.map((x, i) => (typeof x === "object" ? x.date ?? x.x ?? i : i));
    const fd = d.forecast_dates || res.forecast_dates || fc.map((_, i) => `+${i + 1}`);
    fc = fc.map((x) => (typeof x === "object" ? x.value ?? x.y : x));
    return { hv, hd, fc, fd };
  }
  function seriesSub(res) {
    const { fc } = seriesData(res);
    return fc.length > 1 ? `Dự báo ${fc.length} bước tới; bước đầu tiên` : "Dự báo bước tới";
  }
  function drawChart(res) {
    if (!window.Chart) return;
    const { hv, hd, fc, fd } = seriesData(res);
    const tail = Math.min(hv.length, 120);
    const H = hv.slice(-tail), HD = hd.slice(-tail);
    const labels = [...HD, ...fd];
    const histData = [...H, ...fc.map(() => null)];
    const fcData = [...H.map((_, i) => (i === H.length - 1 ? H[i] : null)), ...fc];
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 13;
    new Chart($("#chart"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Lịch sử", data: histData, borderColor: "#64748b", backgroundColor: "rgba(100,116,139,.12)", fill: true, pointRadius: 0, borderWidth: 2, tension: .25 },
        { label: "Dự báo", data: fcData, borderColor: COLOR, backgroundColor: COLOR, pointRadius: 3, borderWidth: 3, borderDash: [6, 4], tension: .25 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top" } },
        scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } }, y: { ticks: { callback: (v) => fmt(v) } } } },
    });
  }

  // ---- đồ thị
  function renderNeighbors(res, fd) {
    const d = res.details || {};
    const nb = d.neighbors || res.neighbors || [];
    if (!nb.length) return "";
    const items = nb.map((n) => (typeof n === "object" ? n : { id: n }));
    const R = 95, cx = 150, cy = 125, N = Math.min(items.length, 14);
    let svg = `<svg class="graphviz" viewBox="0 0 300 250">`;
    for (let i = 0; i < N; i++) {
      const a = (2 * Math.PI * i) / N - Math.PI / 2, x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
      const same = items[i].same ?? (items[i].label !== undefined && items[i].label === res.label);
      svg += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#cbd5e1" stroke-width="2"/>`;
      svg += `<circle cx="${x}" cy="${y}" r="13" fill="${same ? COLOR : "#94a3b8"}" opacity=".9"/><text x="${x}" y="${y + 4}" font-size="9" text-anchor="middle" fill="#fff">${esc(items[i].id)}</text>`;
    }
    svg += `<circle cx="${cx}" cy="${cy}" r="22" fill="#0f172a"/><text x="${cx}" y="${cy + 4}" font-size="11" text-anchor="middle" fill="#fff">${esc(d.node_id ?? res.node_id ?? fd.get("node_id") ?? "nút")}</text></svg>`;
    // nhãn cho đúng các nút đã vẽ; phần còn lại gộp một chip để khung kết quả không quá dài
    const chips = items.slice(0, N).map((n) => `<span class="nb ${n.same ?? n.label === res.label ? "same" : ""}">#${esc(n.id)}${n.label !== undefined ? " · " + esc(n.label) : ""}</span>`).join("")
      + (items.length > N ? `<span class="nb">+${items.length - N} nút khác</span>` : "");
    return `<h3 style="margin-top:18px">Láng giềng (${items.length})</h3>${svg}<div class="neighbors">${chips}</div>`;
  }

  // ---- hội thoại
  function renderChat(res, fd) {
    const q = fd.get("question") || fd.get("text") || [...fd.values()].find((v) => typeof v === "string") || "";
    const ans = esc(res.answer ?? res.label ?? "").replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
    let html = `<div class="chat"><div class="msg user">${esc(q)}</div><div class="msg bot">${ans}</div></div>`;
    const tools = res.tools || (res.details || {}).tools || [];
    if (tools.length) html += `<div class="toolchips">${tools.map((t) => `<span>🛠 ${esc(typeof t === "object" ? t.name ?? JSON.stringify(t) : t)}</span>`).join("")}</div>`;
    const src = res.sources || [];
    if (src.length) html += `<h3 style="margin-top:16px">Nguồn trích dẫn</h3><div class="sources">${src.slice(0, 6).map((s, i) => {
      if (typeof s !== "object") return `<div class="src"><b>[${i + 1}]</b> ${esc(s)}</div>`;
      const txt = s.text ?? s.content ?? s.doc ?? JSON.stringify(s);
      const sc = s.score !== undefined ? ` <small>(điểm ${fmt(s.score, 3)})</small>` : "";
      const t = String(txt);
      return `<div class="src"><b>[${esc(s.id ?? i + 1)}]</b> ${esc(s.source ?? s.title ?? "")} ${esc(t.length > 260 ? t.slice(0, 260).trimEnd() + "…" : t)}${sc}</div>`;
    }).join("")}</div>`;
    const tr = res.trace || (res.details || {}).trace || [];
    if (tr.length) html += `<details class="trace" open><summary>Vết suy luận (${tr.length} bước)</summary>${tr.map((s) => {
      const [lab, txt] = typeof s === "object" ? [s.step ?? s.type ?? s.role ?? "", s.text ?? s.content ?? s.detail ?? JSON.stringify(s)] : ["", s];
      const t = String(txt).trim();
      return `<div class="step"><span class="lab ${esc(String(lab).toLowerCase())}">${esc(lab || "bước")}</span><span class="txt">${esc(t.length > 400 ? t.slice(0, 400).trimEnd() + "…" : t)}</span></div>`;
    }).join("")}</details>`;
    return html + scalarRows(res.details || {});
  }
})();
