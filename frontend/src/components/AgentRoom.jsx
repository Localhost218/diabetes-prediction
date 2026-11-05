import { useEffect, useMemo, useState } from 'react'
import {
  X, Play, Pause, SkipBack, SkipForward, User, Network, MessageSquareText, FileSearch, HelpCircle,
  Brain, Database, Scale, Dumbbell, PenLine,
} from 'lucide-react'
import './AgentRoom.css'

// Bố cục bám theo sơ đồ trên bảng: DB → DL ở trên, KB (Neo4j) + RAG ở dưới, hội đồng (+/−) bên phải,
// RL + luật ở góc dưới phải, tầng tiếp nhận bên trái; Orchestrator ở giữa điều phối qua MessageBus.
const AGENTS = {
  User:                 { x: 70,  y: 42,  r: 20, label: 'Người dùng',     sub: '',               icon: User,              kind: 'user' },
  ConversationAgent:    { x: 70,  y: 140, r: 24, label: 'Hội thoại',      sub: 'text → hồ sơ',   icon: MessageSquareText, kind: 'intake' },
  DocumentAgent:        { x: 70,  y: 225, r: 24, label: 'Đọc bệnh án',    sub: 'PDF · ảnh · CSV', icon: FileSearch,       kind: 'intake' },
  ClarificationAgent:   { x: 70,  y: 310, r: 24, label: 'Làm rõ',         sub: 'phạm vi · hỏi lại', icon: HelpCircle,     kind: 'intake' },
  Orchestrator:         { x: 240, y: 225, r: 34, label: 'Điều phối',      sub: 'Orchestrator',   icon: Network,           kind: 'core' },
  PredictionAgent:      { x: 240, y: 72,  r: 28, label: 'Dự đoán',        sub: 'NHANES + BRFSS 3,2 tr',   icon: Database,          kind: 'dl' },
  KnowledgeAgent:       { x: 240, y: 378, r: 28, label: 'Tri thức',       sub: 'Neo4j KB + RAG', icon: Brain,             kind: 'kb' },
  ConsensusCriticAgent: { x: 405, y: 225, r: 38, label: 'Hội đồng',       sub: 'tranh biện (+/−)', icon: Scale,           kind: 'critic' },
  TreatmentRLAgent:     { x: 405, y: 378, r: 26, label: 'Can thiệp',      sub: 'RL + luật',      icon: Dumbbell,          kind: 'rl' },
  ResponseAgent:        { x: 405, y: 72,  r: 26, label: 'Phản hồi',       sub: 'tư vấn · trợ giúp', icon: PenLine,         kind: 'resp' },
}

// Các kênh có thể xảy ra (vẽ mờ làm nền)
const CHANNELS = [
  ['User', 'Orchestrator'], ['ResponseAgent', 'User'],
  ...['ConversationAgent', 'DocumentAgent', 'ClarificationAgent', 'PredictionAgent', 'KnowledgeAgent',
    'ConsensusCriticAgent', 'TreatmentRLAgent', 'ResponseAgent'].map(a => ['Orchestrator', a]),
  ['ConsensusCriticAgent', 'PredictionAgent'], ['ConsensusCriticAgent', 'KnowledgeAgent'],
]

const PERF = {
  REQUEST: { label: 'REQUEST', color: 'var(--ar-request)' },
  INFORM: { label: 'INFORM', color: 'var(--ar-inform)' },
  CHALLENGE: { label: 'CHALLENGE', color: 'var(--ar-challenge)' },
  RESPONSE: { label: 'RESPONSE', color: 'var(--ar-response)' },
  FAILURE: { label: 'FAILURE', color: 'var(--ar-failure)' },
}

const RISK = { very_high: 'Rất cao', high: 'Cao', moderate: 'Trung bình', low: 'Thấp' }

const INTENT = {
  RISK_ASSESSMENT: 'Đánh giá nguy cơ', KNOWLEDGE_QUESTION: 'Hỏi kiến thức', FOLLOW_UP: 'Hỏi tiếp',
  SMALL_TALK: 'Trò chuyện', OUT_OF_SCOPE: 'Ngoài phạm vi',
}

// Cung cong giữa hai nút; đảo chiều thì cong sang phía kia để REQUEST và INFORM không chồng nhau
function arc(a, b, bend = 0.18) {
  const A = AGENTS[a], B = AGENTS[b]
  if (!A || !B) return null
  const dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy) || 1
  const ux = dx / len, uy = dy / len
  const sx = A.x + ux * A.r, sy = A.y + uy * A.r
  const ex = B.x - ux * (B.r + 4), ey = B.y - uy * (B.r + 4)
  const mx = (sx + ex) / 2 - uy * len * bend, my = (sy + ey) / 2 + ux * len * bend
  return `M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}`
}

// Thêm hai bước giả "người dùng gửi" và "trả lời người dùng" để vòng hội chẩn khép kín trên sơ đồ
function buildSteps(data) {
  const msgs = (data?.transcript || []).map(m => ({ ...m }))
  if (!msgs.length) return []
  const conclusion = data.consensus
    ? `${data.consensus.risk_label || RISK[data.consensus.risk_level] || ''}`
    : 'Câu trả lời đã gửi trong khung chat.'
  return [
    { sender: 'User', receiver: 'Orchestrator', performative: 'REQUEST', msg_type: 'USER_MESSAGE',
      summary: 'Gửi tin nhắn / tệp bệnh án.', synthetic: true },
    ...msgs,
    { sender: 'ResponseAgent', receiver: 'User', performative: 'INFORM', msg_type: 'ANSWER',
      summary: conclusion, synthetic: true },
  ]
}

function AgentRoom({ message, loading, onClose }) {
  const data = message?.data
  const steps = useMemo(() => buildSteps(data), [data])
  const [step, setStep] = useState(0)
  const [playRequested, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const playing = playRequested && step < steps.length - 1  // tự dừng ở bước cuối

  // Tin nhắn mới → component được dựng lại (key ở App) nên phát lại từ đầu
  useEffect(() => {
    if (!playing) return
    const t = setTimeout(() => setStep(s => s + 1), 1300 / speed)
    return () => clearTimeout(t)
  }, [playing, step, speed])

  const cur = steps[step]
  const seen = useMemo(() => {
    const s = new Set()
    steps.slice(0, step + 1).forEach(m => { s.add(m.sender); s.add(m.receiver) })
    return s
  }, [steps, step])
  const debate = steps.filter(m => m.performative === 'CHALLENGE').length
  const c = data?.consensus

  return (
    <aside className="agent-room" aria-label="Phòng hội chẩn đa tác tử">
      <div className="ar-header">
        <div>
          <div className="ar-title">Phòng hội chẩn đa tác tử</div>
          <div className="ar-subtitle">
            {loading ? 'Các tác tử đang làm việc…'
              : steps.length ? `${INTENT[data.intent] || data.intent} · ${steps.length - 2} thông điệp qua MessageBus`
                + (debate ? ` · ${debate} lượt chất vấn` : '')
                : 'Gửi một tin nhắn để xem các tác tử phối hợp'}
          </div>
        </div>
        <button className="icon-btn" onClick={onClose} title="Đóng"><X size={18} /></button>
      </div>

      <svg className={`ar-stage${loading ? ' is-loading' : ''}`} viewBox="0 0 475 440" role="img"
        aria-label="Sơ đồ các tác tử và thông điệp giữa chúng">
        <defs>
          {Object.entries(PERF).map(([k, v]) => (
            <marker key={k} id={`ar-arrow-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6"
              orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill={v.color} />
            </marker>
          ))}
        </defs>

        {/* Vùng nền theo sơ đồ trên bảng */}
        <rect className="ar-zone" x="18" y="100" width="104" height="282" rx="14" />
        <text className="ar-zone-label" x="70" y="374" textAnchor="middle">Tiếp nhận</text>
        <circle className="ar-zone ar-zone-critic" cx="405" cy="225" r="62" />

        {CHANNELS.map(([a, b]) => <path key={a + b} d={arc(a, b, 0)} className="ar-channel" />)}

        {cur && !loading && (() => {
          const d = arc(cur.sender, cur.receiver)
          const p = PERF[cur.performative] || PERF.INFORM
          return d && (
            <g key={step}>
              <path d={d} className="ar-flow" stroke={p.color} markerEnd={`url(#ar-arrow-${cur.performative})`} />
              <circle r="5" fill={p.color}>
                <animateMotion dur={`${0.7 / speed}s`} path={d} fill="freeze" />
              </circle>
            </g>
          )
        })()}

        {Object.entries(AGENTS).map(([name, a]) => {
          const Icon = a.icon
          const active = !loading && cur && (cur.sender === name || cur.receiver === name)
          const speaking = !loading && cur?.sender === name
          const state = loading ? (name === 'Orchestrator' ? 'active' : 'idle')
            : active ? 'active' : seen.has(name) ? 'done' : 'idle'
          const failed = !loading && cur?.sender === name && cur.performative === 'FAILURE'
          return (
            <g key={name} className={`ar-node ar-${a.kind} is-${state}${speaking ? ' is-speaking' : ''}${failed ? ' is-failed' : ''}`}>
              <circle cx={a.x} cy={a.y} r={a.r} className="ar-node-bg" />
              <Icon x={a.x - a.r * 0.45} y={a.y - a.r * 0.45} width={a.r * 0.9} height={a.r * 0.9} className="ar-node-icon" />
              <text x={a.x} y={a.y + a.r + 13} textAnchor="middle" className="ar-node-label">{a.label}</text>
              {a.sub && <text x={a.x} y={a.y + a.r + 24} textAnchor="middle" className="ar-node-sub">{a.sub}</text>}
            </g>
          )
        })}
      </svg>

      {!loading && cur && (
        <div className="ar-bubble" style={{ borderColor: (PERF[cur.performative] || PERF.INFORM).color }}>
          <div className="ar-bubble-head">
            <strong>{AGENTS[cur.sender]?.label || cur.sender}</strong>
            <span className="ar-arrow">→</span>
            <strong>{AGENTS[cur.receiver]?.label || cur.receiver}</strong>
            <span className="ar-perf" style={{ background: (PERF[cur.performative] || PERF.INFORM).color }}>
              {cur.performative}
            </span>
          </div>
          <div className="ar-bubble-text">{cur.summary || cur.msg_type}</div>
          <div className="ar-bubble-meta">
            <code>{cur.msg_type}</code>{cur.elapsed_ms != null && ` · ${cur.elapsed_ms} ms`}
          </div>
        </div>
      )}

      {steps.length > 0 && !loading && (
        <div className="ar-controls">
          <button className="icon-btn" onClick={() => { setPlaying(false); setStep(s => Math.max(0, s - 1)) }} title="Bước trước">
            <SkipBack size={16} />
          </button>
          <button className="icon-btn ar-play" onClick={() => {
            if (step >= steps.length - 1) { setStep(0); setPlaying(true) } else setPlaying(!playing)
          }} title={playing ? 'Tạm dừng' : 'Phát lại'}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button className="icon-btn" onClick={() => { setPlaying(false); setStep(s => Math.min(steps.length - 1, s + 1)) }} title="Bước sau">
            <SkipForward size={16} />
          </button>
          <input type="range" min="0" max={steps.length - 1} value={step}
            onChange={e => { setPlaying(false); setStep(Number(e.target.value)) }} aria-label="Bước" />
          <span className="ar-count">{step + 1}/{steps.length}</span>
          <button className="ar-speed" onClick={() => setSpeed(s => (s >= 2 ? 0.5 : s * 2))} title="Tốc độ">×{speed}</button>
        </div>
      )}

      {c && !loading && (
        <div className="ar-result">
          <span className={`risk-badge risk-${c.risk_level.replace('_', '-')}`}>Nguy cơ {RISK[c.risk_level]}</span>
          <span className="ar-ev plus">+{c.positive_evidence_plus.length}</span>
          <span className="ar-ev minus">−{c.negative_contradictions_minus.length}</span>
          {c.safety_guardrails.length > 0 && <span className="ar-ev guard">⚠ {c.safety_guardrails.length} cảnh báo</span>}
          {data.treatment?.recommended_action && <span className="ar-ev rl">RL: {data.treatment.recommended_action}</span>}
        </div>
      )}

      {steps.length > 0 && !loading && (
        <ol className="ar-log">
          {steps.map((m, i) => (
            <li key={i} className={`${i === step ? 'is-current' : ''} ${i > step ? 'is-future' : ''} perf-${m.performative.toLowerCase()}`}
              onClick={() => { setPlaying(false); setStep(i) }}>
              <span className="ar-log-dot" style={{ background: (PERF[m.performative] || PERF.INFORM).color }} />
              <span className="ar-log-who">{AGENTS[m.sender]?.label || m.sender} → {AGENTS[m.receiver]?.label || m.receiver}</span>
              <span className="ar-log-text">{m.summary || m.msg_type}</span>
            </li>
          ))}
        </ol>
      )}

      <div className="ar-legend">
        {Object.entries(PERF).map(([k, v]) => (
          <span key={k}><i style={{ background: v.color }} />{v.label}</span>
        ))}
      </div>
    </aside>
  )
}

export default AgentRoom
