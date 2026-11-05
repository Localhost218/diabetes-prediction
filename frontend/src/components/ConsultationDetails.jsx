import { useState } from 'react'
import { ChevronDown, ChevronRight, Network, Swords, BookOpen, Activity, Workflow } from 'lucide-react'

const RISK_STYLE = {
  very_high: { label: 'Rất cao', className: 'risk-very-high' },
  high: { label: 'Cao', className: 'risk-high' },
  moderate: { label: 'Trung bình', className: 'risk-moderate' },
  low: { label: 'Thấp', className: 'risk-low' },
}

const SHORT = {
  Orchestrator: 'Orchestrator', ConversationAgent: 'Conversation', DocumentAgent: 'Document',
  PredictionAgent: 'Prediction', KnowledgeAgent: 'Knowledge', ConsensusCriticAgent: 'Critic',
  TreatmentRLAgent: 'TreatmentRL', ResponseAgent: 'Response',
}

function Section({ icon, title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="details-section">
      <button className="details-section-header" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {icon}
        <span>{title}</span>
      </button>
      {open && <div className="details-section-body">{children}</div>}
    </div>
  )
}

function ConsultationDetails({ data }) {
  const [open, setOpen] = useState(false)
  if (!data || data.error || !data.transcript) return null
  const c = data.consensus
  const risk = c ? RISK_STYLE[c.risk_level] : null

  return (
    <div className="consultation-details">
      <button className="details-toggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {c ? 'Chi tiết hội chẩn đa tác tử' : 'Luồng xử lý của các tác tử'}
        {risk && <span className={`risk-badge ${risk.className}`}>Nguy cơ {risk.label}</span>}
        {c && <span className="details-meta">{c.debate_rounds} vòng tranh biện · {data.transcript.length} thông điệp</span>}
      </button>

      {open && (
        <div className="details-body">
          {c && (
            <div className="consensus-circle">
              <div className="consensus-title">Vòng tròn hội đồng (+ / −)</div>
              {c.safety_guardrails.map((g, i) => (
                <div key={`g${i}`} className={`evidence guardrail guardrail-${g.level}`}>⚠️ {g.message}</div>
              ))}
              {c.positive_evidence_plus.map((p, i) => (
                <div key={`p${i}`} className="evidence plus">{p.replace(/^\[\+\]\s*/, '')}</div>
              ))}
              {c.negative_contradictions_minus.map((m, i) => (
                <div key={`m${i}`} className="evidence minus">{m.replace(/^\[-\]\s*/, '')}</div>
              ))}
            </div>
          )}

          {c && c.debate.length > 0 && (
            <Section icon={<Swords size={14} />} title={`Tranh biện (${c.debate_rounds} vòng)`} defaultOpen>
              {c.debate.map((d, i) => (
                <div key={i} className="debate-item">
                  <div className="debate-head">
                    <span className="debate-round">Vòng {d.round}</span>
                    Critic → <strong>{SHORT[d.challenged] || d.challenged}</strong>
                    <code>{d.challenge}</code>
                  </div>
                  {d.statement && <div className="debate-statement">{d.statement}</div>}
                </div>
              ))}
            </Section>
          )}

          {data.ml && (
            <Section icon={<Activity size={14} />} title="Nhánh DB → ML/DL: dự đoán">
              <div className="kv">
                Xác suất theo yếu tố nguy cơ: <strong>{(data.ml.risk_probability * 100).toFixed(1)}%</strong>
                {' '}(gấp {data.ml.relative_risk} lần trung bình) · dữ liệu đầy đủ {(data.ml.data_completeness * 100).toFixed(0)}%
              </div>
              {data.ml.models && (
                <div className="model-compare">
                  {['ml', 'dl'].map(k => {
                    const m = data.ml.models[k]
                    const metrics = data.ml.model_metrics[k]
                    return (
                      <div key={k} className="model-card">
                        <div className="model-kind">{m.kind}</div>
                        <div className="model-name">{m.name}</div>
                        <div className="model-prob">{(m.risk_probability * 100).toFixed(1)}%</div>
                        <div className="kv muted">AUC kiểm tra {metrics.roc_auc} · khi thiếu dữ liệu {metrics.partial_data_auc}</div>
                      </div>
                    )
                  })}
                </div>
              )}
              {data.ml.large_scale && (
                <div className="model-card large-scale">
                  <div className="model-kind">Ý kiến thứ hai · quy mô lớn</div>
                  <div className="model-name">{data.ml.large_scale.dataset}</div>
                  <div className="model-prob">{(data.ml.large_scale.risk_probability * 100).toFixed(1)}%</div>
                  <div className="kv muted">
                    gấp {data.ml.large_scale.relative_risk} lần trung bình (tỷ lệ nền {(data.ml.large_scale.population_base_rate * 100).toFixed(1)}%)
                    · ML {(data.ml.large_scale.models.ml * 100).toFixed(0)}% · DL {(data.ml.large_scale.models.dl * 100).toFixed(0)}%
                    {data.ml.large_scale.roc_auc?.dl && ` · AUC ${data.ml.large_scale.roc_auc.dl}`}
                  </div>
                </div>
              )}
              <div className="kv muted">
                Dữ liệu huấn luyện: {data.ml.dataset} · {data.ml.model_metrics.ml?.n_train?.toLocaleString()} mẫu
                {data.ml.disagreement != null && ` · độ lệch ML–DL ${(data.ml.disagreement * 100).toFixed(1)} điểm %`}
              </div>
              {data.ml.top_risk_factors.map((f, i) => (
                <div key={i} className="factor-row">
                  <span>{f.feature_vi}</span>
                  <span className="factor-bar">
                    <span className={f.impact > 0 ? 'bar-up' : 'bar-down'}
                      style={{ width: `${Math.min(100, Math.abs(f.impact) * 400)}%` }} />
                  </span>
                  <span className="factor-val">{f.impact > 0 ? '+' : ''}{(f.impact * 100).toFixed(1)}%</span>
                </div>
              ))}
            </Section>
          )}

          {(data.graph_paths?.length > 0 || data.citations?.length > 0) && (
            <Section icon={<Network size={14} />} title="Nhánh KB → RAG: đồ thị tri thức & nguồn">
              {data.graph_paths?.map((p, i) => <div key={i} className="graph-path">{p}</div>)}
              {data.citations?.length > 0 && (
                <div className="citations">
                  <BookOpen size={13} />
                  {data.citations.map((x, i) => (
                    <span key={i} className="citation">{x.title} — {x.section}</span>
                  ))}
                </div>
              )}
            </Section>
          )}

          <Section icon={<Workflow size={14} />} title={`Kế hoạch & thông điệp (${data.transcript.length})`}>
            <div className="plan">{data.plan.map((p, i) => <span key={i} className="plan-step">{p}</span>)}</div>
            <div className="transcript">
              {data.transcript.map((m, i) => (
                <div key={i} className={`transcript-row perf-${m.performative.toLowerCase()}`}>
                  <span className="t-from">{SHORT[m.sender] || m.sender}</span>
                  <span className="t-arrow">→</span>
                  <span className="t-to">{SHORT[m.receiver] || m.receiver}</span>
                  <span className="t-perf">{m.performative}</span>
                  <code className="t-type">{m.msg_type}</code>
                  {m.elapsed_ms != null && <span className="t-ms">{m.elapsed_ms} ms</span>}
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}

export default ConsultationDetails
