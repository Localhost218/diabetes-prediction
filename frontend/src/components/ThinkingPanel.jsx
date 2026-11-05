import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, BrainCircuit, Network } from 'lucide-react'
import './ThinkingPanel.css'

// Tên hiển thị ngắn của các tác tử (giống Phòng hội chẩn)
const AGENT_VI = {
  User: 'Người dùng', Orchestrator: 'Điều phối', ConversationAgent: 'Hội thoại', DocumentAgent: 'Đọc bệnh án',
  ClarificationAgent: 'Làm rõ', PredictionAgent: 'Dự đoán', KnowledgeAgent: 'Tri thức',
  ConsensusCriticAgent: 'Hội đồng', TreatmentRLAgent: 'Can thiệp', ResponseAgent: 'Phản hồi', MessageBus: 'Bus',
}

const LABEL_COLOR = {
  Disease: 'var(--kg-disease)', RiskFactor: 'var(--kg-risk)', Test: 'var(--kg-test)', Drug: 'var(--kg-drug)',
  Intervention: 'var(--kg-intervention)', Complication: 'var(--kg-complication)',
}
const LABEL_VI = {
  Disease: 'Bệnh', RiskFactor: 'Yếu tố nguy cơ', Test: 'Xét nghiệm', Drug: 'Thuốc', Intervention: 'Can thiệp',
  Complication: 'Biến chứng',
}
const RELATION_VI = {
  HAS_RISK_FACTOR: 'có yếu tố nguy cơ', DIAGNOSED_BY: 'chẩn đoán bằng', CAUSES: 'gây', SCREENED_BY: 'tầm soát bằng',
  TREATED_BY: 'điều trị bằng', CONTRAINDICATED_IN: 'chống chỉ định khi', PREVENTED_BY: 'phòng ngừa bằng',
}

/** Bố cục xuyên tâm: nút xuất phát ở giữa, nút cách 1 bước ở vòng trong, 2 bước ở vòng ngoài. */
function layout(graph, w, h) {
  const start = new Set(graph.entities || [])
  const nodes = new Map()
  const add = (name, label, hop) => {
    const cur = nodes.get(name)
    const ring = start.has(name) ? 0 : hop
    if (!cur || ring < cur.ring) nodes.set(name, { name, label: label || cur?.label, ring })
  }
  for (const e of graph.edges || []) {
    add(e.head, e.head_label, e.hops)
    add(e.tail, e.tail_label, e.hops)
  }
  const rings = [[], [], []]
  for (const n of nodes.values()) rings[Math.min(2, n.ring)].push(n)
  const cx = w / 2, cy = h / 2
  const radius = [rings[0].length > 1 ? 34 : 0, Math.min(w, h) * 0.30, Math.min(w, h) * 0.45]
  rings.forEach((ring, r) => ring.forEach((n, i) => {
    const a = (2 * Math.PI * i) / Math.max(1, ring.length) + r * 0.4 - Math.PI / 2
    n.x = cx + radius[r] * Math.cos(a) * (w > h ? 1.45 : 1)
    n.y = cy + radius[r] * Math.sin(a)
  }))
  return { nodes: [...nodes.values()], cx, cy }
}

export function KnowledgeGraphView({ graph }) {
  const W = 520, H = 280
  const { nodes } = useMemo(() => layout(graph, W, H), [graph])
  const pos = useMemo(() => Object.fromEntries(nodes.map(n => [n.name, n])), [nodes])
  const labels = [...new Set(nodes.map(n => n.label).filter(Boolean))]
  return (
    <div className="kg-view">
      <div className="kg-title"><Network size={13} /> Đồ thị tri thức (Neo4j) mà agent Tri thức vừa duyệt
        <span className="kg-count">{nodes.length} nút · {graph.edges.length} cạnh</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="kg-svg" role="img" aria-label="Đồ thị tri thức đã duyệt">
        {graph.edges.map((e, i) => {
          const a = pos[e.head], b = pos[e.tail]
          if (!a || !b) return null
          return (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              className={`kg-edge hop-${e.hops}`} style={{ animationDelay: `${0.15 + i * 0.04}s` }}>
              <title>{`${e.head} —${RELATION_VI[e.relation] || e.relation}→ ${e.tail}`}</title>
            </line>
          )
        })}
        {nodes.map((n, i) => (
          <g key={n.name} className={`kg-node ring-${n.ring}`} style={{ animationDelay: `${i * 0.05}s` }}>
            <circle cx={n.x} cy={n.y} r={n.ring === 0 ? 9 : 6} fill={LABEL_COLOR[n.label] || 'var(--text-secondary)'} />
            <text x={n.x} y={n.y - (n.ring === 0 ? 13 : 10)} textAnchor="middle">
              {n.name.length > 22 ? n.name.slice(0, 21) + '…' : n.name}
            </text>
            <title>{`${n.name} (${LABEL_VI[n.label] || n.label})`}</title>
          </g>
        ))}
      </svg>
      <div className="kg-legend">
        {labels.map(l => <span key={l}><i style={{ background: LABEL_COLOR[l] }} />{LABEL_VI[l] || l}</span>)}
      </div>
    </div>
  )
}

/**
 * Khối "suy luận" của hội đồng trong khung chat.
 * - live: đang chạy → mở sẵn, hiện từng thông điệp khi nó tới, có đồ thị khi agent Tri thức duyệt Neo4j.
 * - xong: thu gọn thành dropdown "Đã suy luận qua N bước", bấm mới mở.
 */
function ThinkingPanel({ steps = [], graph = null, live = false, seconds = null }) {
  const [open, setOpen] = useState(false)
  const listRef = useRef(null)
  const expanded = live || open

  useEffect(() => {
    if (live && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [live, steps.length])

  if (!live && steps.length === 0) return null
  const challenges = steps.filter(s => s.performative === 'CHALLENGE').length
  const last = steps[steps.length - 1]

  return (
    <div className={`thinking-panel${live ? ' is-live' : ''}`}>
      <button className="tp-header" onClick={() => !live && setOpen(o => !o)} disabled={live} aria-expanded={expanded}>
        <BrainCircuit size={15} className="tp-icon" />
        {live ? (
          <span className="tp-live-text">
            Hội đồng đang suy luận<span className="tp-dots"><i>.</i><i>.</i><i>.</i></span>
            {last && <span className="tp-current"> {AGENT_VI[last.sender] || last.sender} → {AGENT_VI[last.receiver] || last.receiver}</span>}
          </span>
        ) : (
          <span>
            Đã suy luận qua {steps.length} bước{challenges > 0 && ` · ${challenges} lượt chất vấn`}
            {graph?.edges?.length > 0 && ' · có duyệt đồ thị tri thức'}{seconds != null && ` · ${seconds} giây`}
          </span>
        )}
        {!live && (open ? <ChevronDown size={15} /> : <ChevronRight size={15} />)}
      </button>

      {expanded && (
        <div className="tp-body">
          <ol className="tp-steps" ref={listRef}>
            {steps.map((s, i) => (
              <li key={i} className={`perf-${s.performative.toLowerCase()}`}>
                <span className="tp-who">{AGENT_VI[s.sender] || s.sender} → {AGENT_VI[s.receiver] || s.receiver}</span>
                <span className="tp-perf">{s.performative}</span>
                <span className="tp-text">{s.summary || s.msg_type}</span>
              </li>
            ))}
            {live && <li className="tp-pending"><span className="tp-who">…</span></li>}
          </ol>
          {graph?.edges?.length > 0 && <KnowledgeGraphView graph={graph} />}
        </div>
      )}
    </div>
  )
}

export default ThinkingPanel
