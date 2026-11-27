// Hàm thuần cho bố cục "Hội chẩn": đọc hồ sơ, đối chiếu ngưỡng, suy ra tiến trình lượt và tác tử đang chạy.

// "Tuổi: 52 tuổi (bạn cung cấp)" → { label, value, source, tag, num }
export function parseProfileLine(line) {
  const colon = line.indexOf(':')
  const rest = line.slice(colon + 1).trim()
  const m = rest.match(/^(.*?)\s*\(([^)]*)\)\s*$/)
  const value = m ? m[1] : rest
  const source = m ? m[2] : ''
  const tag = /^từ file/.test(source) ? 'file' : /tính/.test(source) ? 'tính' : /form/.test(source) ? 'form' : 'chat'
  const num = parseFloat(value.replace(',', '.'))
  return { label: line.slice(0, colon).trim(), value, source, tag, num: Number.isFinite(num) ? num : null }
}

// Mỗi nhóm có các mục "nên có" để tính độ đầy đủ; mục khác vẫn hiện nếu người dùng đã cung cấp.
export const PROFILE_GROUPS = [
  { key: 'about', title: 'Về bạn', expected: ['Tuổi', 'Giới tính', 'BMI', 'Vòng eo'],
    match: /^(Tuổi|Giới tính|Chiều cao|Cân nặng|BMI|Vòng eo)$/ },
  { key: 'labs', title: 'Xét nghiệm', expected: ['Đường huyết lúc đói', 'HbA1c', 'Huyết áp tâm thu', 'Cholesterol toàn phần', 'eGFR'],
    match: /^(Đường huyết|Glucose 2 giờ|HbA1c|Huyết áp|Cholesterol|LDL|HDL|Triglycerid|Creatinin|eGFR|Chuỗi đường huyết)/ },
  { key: 'diet', title: 'Ăn uống & giấc ngủ', expected: ['Ăn rau xanh mỗi ngày', 'Ăn trái cây mỗi ngày', 'Giờ ngủ mỗi đêm'],
    match: /^(Ăn rau xanh|Ăn trái cây|Giờ ngủ)/ },
  { key: 'life', title: 'Lối sống & tiền sử', expected: ['Tiền sử gia đình mắc ĐTĐ', 'Tăng huyết áp', 'Hút thuốc', 'Vận động thể lực thường xuyên'],
    match: null },
]

export function groupProfile(summary = []) {
  const items = summary.map(parseProfileLine)
  const groups = PROFILE_GROUPS.map(g => {
    const own = items.filter(it => (g.match ? g.match.test(it.label) : !PROFILE_GROUPS.some(o => o.match?.test(it.label))))
    const missing = g.expected.filter(label => !own.some(it => it.label === label))
    return { ...g, items: own, missing, filled: g.expected.length - missing.length }
  })
  const total = groups.reduce((n, g) => n + g.expected.length, 0)
  const filled = groups.reduce((n, g) => n + g.filled, 0)
  return { groups, filled, total, percent: Math.round((100 * filled) / total) }
}

// Ngưỡng ADA 2024 (mmol/L cho glucose; % cho HbA1c). Hồ sơ lưu glucose theo mg/dL.
const MGDL_PER_MMOL = 18.016
export const THRESHOLDS = [
  { key: 'fpg', name: 'Glucose lúc đói', unit: 'mmol/L', normal: '< 5,6', pre: '5,6–6,9', dm: '≥ 7,0', cut: [5.6, 7.0],
    label: 'Đường huyết lúc đói', toUnit: v => v / MGDL_PER_MMOL },
  { key: 'a1c', name: 'HbA1c', unit: '%', normal: '< 5,7', pre: '5,7–6,4', dm: '≥ 6,5', cut: [5.7, 6.5],
    label: 'HbA1c', toUnit: v => v },
  { key: 'ogtt', name: 'OGTT 2 giờ', unit: 'mmol/L', normal: '< 7,8', pre: '7,8–11,0', dm: '≥ 11,1', cut: [7.8, 11.1],
    label: 'Glucose 2 giờ sau nghiệm pháp (OGTT)', toUnit: v => v / MGDL_PER_MMOL },
]

/** 'normal' | 'pre' | 'dm' | null cho một dòng hồ sơ, dựa trên ngưỡng ADA. */
export function glycemicBand(item) {
  const t = THRESHOLDS.find(x => x.label === item.label)
  if (!t || item.num == null) return null
  const v = t.toUnit(item.num)
  return v >= t.cut[1] ? 'dm' : v >= t.cut[0] ? 'pre' : 'normal'
}

// Bảy bước của một lượt (khớp OrchestratorAgent.run_turn), suy ra từ msg_type của các thông điệp trên bus.
export const PLAN_STAGES = [
  { key: 'intake', label: 'Tiếp nhận', types: ['EXTRACT_FROM_TEXT', 'PARSE_DOCUMENT'] },
  { key: 'intent', label: 'Ý định', types: ['SCOPE_CHECK'] },
  { key: 'clarify', label: 'Làm rõ', types: ['PLAN_QUESTIONS'] },
  { key: 'branch', label: 'ML ‖ RAG', types: ['PREDICT_RISK', 'RETRIEVE_EVIDENCE', 'ANSWER_QUESTION'] },
  { key: 'debate', label: 'Tranh biện', types: ['DELIBERATE'], prefix: 'CHALLENGE_' },
  { key: 'rl', label: 'Gợi ý RL', types: ['RECOMMEND_TREATMENT'] },
  { key: 'respond', label: 'Trả lời', prefix: 'COMPOSE_' },
]

const stageOf = type => PLAN_STAGES.findIndex(s => s.types?.includes(type) || (s.prefix && type?.startsWith(s.prefix)))

/** Trạng thái từng bước: 'done' | 'now' | 'skip' | 'todo'. live = lượt đang chạy. */
export function planStatus(steps = [], live = false) {
  const reached = new Set(steps.map(s => stageOf(s.msg_type)).filter(i => i >= 0))
  if (reached.size) reached.add(1) // phân loại ý định chạy ở Orchestrator, không có thông điệp riêng
  const last = Math.max(-1, ...reached)
  return PLAN_STAGES.map((s, i) => {
    if (!steps.length) return 'todo'
    if (live) return i === last ? 'now' : reached.has(i) ? 'done' : i < last ? 'skip' : 'todo'
    return reached.has(i) ? 'done' : 'skip'
  })
}

/** Tác tử đang được gọi (người nhận của REQUEST/CHALLENGE cuối cùng), hoặc null. */
export function activeAgent(steps = []) {
  for (let i = steps.length - 1; i >= 0; i--) {
    const s = steps[i]
    if (s.performative === 'REQUEST' || s.performative === 'CHALLENGE') return s.receiver
  }
  return null
}
