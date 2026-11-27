import { useEffect, useState } from 'react'
import { Workflow, Database, Bot } from 'lucide-react'
import { activeAgent } from '../services/workspace'
import { getJSON } from '../services/llmService'

// Bản vẽ mạch: mọi thông điệp đi qua thanh MessageBus (bên trái) rồi rẽ vào làn của từng giai đoạn.
const BUS_X = 24
const ZONES = [
  { key: 'intake', label: '01 · Tiếp nhận', y1: 76, y2: 158, lane: 150 },
  { key: 'branch', label: '02 · Song song ML ‖ RAG', y1: 168, y2: 282, lane: 274 },
  { key: 'council', label: '03 · Hội đồng', y1: 292, y2: 364, lane: 356 },
  { key: 'reply', label: '04 · Trả lời', y1: 374, y2: 450, lane: 442 },
]
const NODES = [
  { id: 'Orchestrator', label: 'Orchestrator', x: 92, y: 40, lane: 50, fill: '#cfe3b2', side: true },
  { id: 'ConversationAgent', label: 'Conversation', x: 96, y: 108, lane: 150, fill: '#e8efd9' },
  { id: 'DocumentAgent', label: 'Document', x: 176, y: 108, lane: 150, fill: '#e8efd9' },
  { id: 'ClarificationAgent', label: 'Clarification', x: 258, y: 108, lane: 150, fill: '#e8efd9' },
  { id: 'PredictionAgent', label: 'Prediction', x: 100, y: 200, lane: 274, fill: '#d6e5f6' },
  { id: 'KnowledgeAgent', label: 'Knowledge', x: 196, y: 200, lane: 274, fill: '#e7ddf5' },
  { id: 'ConsensusCriticAgent', label: 'Consensus', x: 118, y: 318, lane: 356, fill: '#f6cfc7' },
  { id: 'TreatmentRLAgent', label: 'TreatmentRL', x: 236, y: 318, lane: 356, fill: '#fcdcbd' },
  { id: 'ResponseAgent', label: 'Response', x: 166, y: 400, lane: 442, fill: '#fbe5bd' },
]
const KB = [{ id: 'qdrant', label: 'Qdrant', x: 284, y: 196 }, { id: 'neo4j', label: 'Neo4j', x: 284, y: 244 }]
const POS = Object.fromEntries(NODES.map(n => [n.id, n]))
const SHORT = Object.fromEntries(NODES.map(n => [n.id, n.label]))
const PERF_CLS = { REQUEST: 'req', INFORM: 'inf', CHALLENGE: 'ch', RESPONSE: 'rs' }
const pill = (text) => text.length * 6 + 12

// Đường đi của một thông điệp: xuống làn của tác tử gửi → thanh bus → làn của tác tử nhận → lên khối nhận
const route = (a, b) => `M${a.x} ${a.y + (a.side ? 8 : 20)} V${a.lane} H${BUS_X} V${b.lane} H${b.x} V${b.y + (b.side ? 8 : 20)}`

function Block({ n, state, count }) {
  const w = pill(n.label), lx = n.side ? n.x + 30 + w / 2 : n.x, ly = n.side ? n.y + 4 : n.y + 34
  return (
    <g className={`fl-node is-${state}`}>
      {state === 'active' && <ellipse className="fl-glow" cx={n.x} cy={n.y + 14} rx="30" ry="12" />}
      <g className="fl-block" transform={`translate(${n.x} ${state === 'active' ? n.y - 5 : n.y})`}>
        <path className="top" d="M0 -11 L22 0 L0 11 L-22 0Z" style={{ fill: n.fill }} />
        <path className="left" d="M-22 0 L0 11 L0 20 L-22 9Z" />
        <path className="right" d="M22 0 L0 11 L0 20 L22 9Z" />
      </g>
      <rect className="fl-pill" x={lx - w / 2} y={ly - 8} width={w} height="16" rx="8" />
      <text className="fl-label" x={lx} y={ly + 3.5} textAnchor="middle">{n.label}</text>
      {count > 0 && (
        <g className="fl-badge" transform={`translate(${n.x + 22} ${n.y - 10})`}>
          <circle r="7.5" /><text y="3" textAnchor="middle">{count > 99 ? '99+' : count}</text>
        </g>
      )}
    </g>
  )
}

function Cylinder({ n, on, note }) {
  return (
    <g className={`fl-kb${on ? ' is-active' : ''}`} transform={`translate(${n.x} ${n.y})`}>
      <path className="side" d="M-15 -7 V9 A15 5.5 0 0 0 15 9 V-7" />
      <ellipse className="lid" cx="0" cy="-7" rx="15" ry="5.5" />
      <text className="fl-label" y="27" textAnchor="middle">{n.label}</text>
      {note && <text className="fl-note" y="37" textAnchor="middle">{note}</text>}
    </g>
  )
}

/**
 * Sidebar phải: luồng thông điệp qua 9 tác tử và KB của lượt đang chạy (hoặc lượt gần nhất).
 * steps: [{sender, receiver, performative, msg_type, summary}] — cùng dạng với khối suy luận trong chat.
 */
function AgentFlow({ steps = [], live = false, graph = null, onOpenKb, onOpenAgents }) {
  const [kb, setKb] = useState(null)
  useEffect(() => { getJSON('/api/kb/stats').then(setKb).catch(() => setKb(null)) }, [])

  const counts = {}
  for (const s of steps) for (const a of [s.sender, s.receiver]) counts[a] = (counts[a] || 0) + 1
  const current = live ? activeAgent(steps) : null
  const last = steps[steps.length - 1]
  const kbOn = current === 'KnowledgeAgent'
  const kbUsed = steps.some(s => s.receiver === 'KnowledgeAgent')
  const stateOf = id => (id === current ? 'active' : counts[id] ? 'done' : steps.length && !live ? 'skip' : 'idle')
  const from = live && last && POS[last.sender], to = live && last && POS[last.receiver]
  const challenges = steps.filter(s => s.performative === 'CHALLENGE').length
  const know = POS.KnowledgeAgent

  return (
    <aside className="nb-flow" aria-label="Luồng xử lý của hệ đa tác tử">
      <section className="rail-card fl-card">
        <h2><Workflow size={17} aria-hidden="true" /> Luồng xử lý
          <span className={`nb-status${live ? ' is-running' : ''}`}>{live ? 'đang chạy' : steps.length ? 'xong' : 'sẵn sàng'}</span></h2>
        <svg className="nb-blueprint fl-svg" viewBox="0 0 320 458" role="img"
          aria-label={live && last ? `Đang chạy: ${last.sender} gửi ${last.performative} ${last.msg_type} tới ${last.receiver}` : 'Sơ đồ 9 tác tử và cơ sở tri thức nối qua MessageBus'}>
          {ZONES.map(z => (
            <g key={z.key} className="fl-zone">
              <rect x="38" y={z.y1} width="276" height={z.y2 - z.y1} rx="10" />
              <text x="46" y={z.y1 + 12}>{z.label}</text>
            </g>
          ))}
          <path className="fl-bus" d={`M${BUS_X} 50 V442`} />
          <text className="fl-bus-label" transform={`translate(${BUS_X - 8} 250) rotate(-90)`} textAnchor="middle">MESSAGEBUS · FIPA-ACL</text>
          <path className="fl-wire" d={[
            `M${BUS_X} 50 H${POS.Orchestrator.x - 22}`,
            ...ZONES.map(z => `M${BUS_X} ${z.lane} H${Math.max(...NODES.filter(n => n.lane === z.lane).map(n => n.x))}`),
            ...NODES.filter(n => !n.side).map(n => `M${n.x} ${n.y + 20} V${n.lane}`),
          ].join(' ')} />
          <path className={`fl-kb-wire${kbOn ? ' is-active' : ''}`} d={KB.map(k => `M${know.x + 22} ${know.y + 4} L${k.x - 15} ${k.y + 2}`).join(' ')} />
          {KB.map(n => <Cylinder key={n.id} n={n} on={kbOn} note={n.id === 'neo4j' && graph?.edges?.length ? `${graph.edges.length} cạnh` : null} />)}
          {NODES.map(n => <Block key={n.id} n={n} state={stateOf(n.id)} count={counts[n.id] || 0} />)}
          {from && to && (
            <g key={steps.length} className={`fl-msg perf-${last.performative.toLowerCase()}`}>
              <path d={route(from, to)} />
              <circle r="4.5"><animateMotion dur="0.9s" fill="freeze" path={route(from, to)} /></circle>
            </g>
          )}
        </svg>
        <div className={`fl-caption${live ? ' is-live' : ''}`}>
          {live && last ? <>
            <span className={`nb-perf p-${PERF_CLS[last.performative] || 'fail'}`}>{last.performative}</span>
            <span><b>{SHORT[last.sender] || last.sender} → {SHORT[last.receiver] || last.receiver}</b><small>{last.msg_type}</small></span>
          </> : steps.length ? <span><b>{steps.length} thông điệp</b><small>{challenges} lượt chất vấn · {kbUsed ? 'có tra KB' : 'không tra KB'}</small></span>
            : <span><b>Chưa chạy</b><small>Gửi câu hỏi hoặc nhập hồ sơ để xem các tác tử phối hợp.</small></span>}
        </div>
      </section>

      <section className="rail-card fl-log-card">
        <h2 className="fl-log-title">Thông điệp gần nhất <span className="nb-count">{steps.length}</span></h2>
        {steps.length === 0 ? <p className="rail-muted">Chưa có thông điệp nào trên MessageBus.</p> : (
          <ol className="fl-log">
            {steps.slice(-6).map((s, i) => (
              <li key={steps.length - 6 + i} className={'perf-' + (PERF_CLS[s.performative] || 'fail')}>
                <div className="fl-log-head">
                  <span className="fl-who">{SHORT[s.sender] || s.sender} → {SHORT[s.receiver] || s.receiver}</span>
                  <span className={`nb-perf p-${PERF_CLS[s.performative] || 'fail'}`}>{s.performative}</span>
                </div>
                <p className="fl-sum">{s.summary || s.msg_type}</p>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section className="rail-card fl-kb-card">
        <h2 className="fl-log-title"><Database size={15} aria-hidden="true" /> Cơ sở tri thức</h2>
        <div className="fl-stats">
          {[['tài liệu', kb?.documents.length], ['đoạn', kb?.chunks], ['nút', kb?.graph.nodes], ['cạnh', kb?.graph.edges]].map(([k, v]) => (
            <div key={k}><strong>{v ?? '–'}</strong><span>{k}</span></div>
          ))}
        </div>
        <div className="fl-actions">
          <button className="btn-secondary" onClick={onOpenKb}><Database size={14} aria-hidden="true" /> Xem KB</button>
          <button className="btn-secondary" onClick={onOpenAgents}><Bot size={14} aria-hidden="true" /> 9 tác tử</button>
        </div>
      </section>
    </aside>
  )
}

export default AgentFlow
