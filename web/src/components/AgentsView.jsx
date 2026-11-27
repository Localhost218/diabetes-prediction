import { Cpu, Gauge, Workflow, MessageSquareText, FileText, HelpCircle, BrainCircuit, BookOpen, Scale, Target, PenLine } from 'lucide-react'

// Vai trò và các loại thông điệp mà từng tác tử xử lý (khớp thuộc tính handles trong src/agents/*.py)
const AGENTS = [
  { name: 'Orchestrator', icon: Workflow, stage: 'core', file: 'orchestrator.py', role: 'Phân loại ý định, lập kế hoạch động, điều phối qua MessageBus', kind: 'Hướng mục tiêu', handles: ['run_turn'] },
  { name: 'ConversationAgent', icon: MessageSquareText, stage: 'intake', file: 'conversation_agent.py', role: 'Trích chỉ số bệnh nhân từ tin nhắn', kind: 'Phản xạ có mô hình', handles: ['EXTRACT_FROM_TEXT'] },
  { name: 'DocumentAgent', icon: FileText, stage: 'intake', file: 'document_agent.py', role: 'Đọc PDF / ảnh / CSV phiếu xét nghiệm', kind: 'Phản xạ có mô hình', handles: ['PARSE_DOCUMENT'] },
  { name: 'ClarificationAgent', icon: HelpCircle, stage: 'intake', file: 'clarification_agent.py', role: 'Kiểm tra phạm vi, đặt câu hỏi làm rõ', kind: 'Hướng mục tiêu', handles: ['SCOPE_CHECK', 'PLAN_QUESTIONS'] },
  { name: 'PredictionAgent', icon: BrainCircuit, stage: 'predict', file: 'prediction_agent.py', role: 'Ensemble GBDT + MLP (NHANES), ý kiến thứ hai BRFSS, LSTM, CNN', kind: 'Dùng mô hình đã học', handles: ['PREDICT_RISK', 'CHALLENGE_ML_VS_LABS', 'CHALLENGE_DATA_QUALITY', 'CHALLENGE_MODEL_DISAGREEMENT', 'CHALLENGE_COHORT_DISAGREEMENT'] },
  { name: 'KnowledgeAgent', icon: BookOpen, stage: 'knowledge', file: 'knowledge_agent.py', role: 'Hybrid RAG (Qdrant + FTS) + GraphRAG (Neo4j)', kind: 'Suy diễn (deliberative)', handles: ['RETRIEVE_EVIDENCE', 'ANSWER_QUESTION', 'CHALLENGE_VERIFY_CRITERIA', 'CHALLENGE_SCREENING_PLAN'] },
  { name: 'ConsensusCriticAgent', icon: Scale, stage: 'council', file: 'consensus_critic_agent.py', role: 'Hội đồng tranh biện +/−, chất vấn ngược các tác tử', kind: 'Hướng tiện ích', handles: ['DELIBERATE'] },
  { name: 'TreatmentRLAgent', icon: Target, stage: 'rl', file: 'treatment_rl_agent.py', role: 'Gợi ý can thiệp bằng Q-learning (24 trạng thái, γ = 0,9)', kind: 'Tác tử học', handles: ['RECOMMEND_TREATMENT'] },
  { name: 'ResponseAgent', icon: PenLine, stage: 'reply', file: 'response_agent.py', role: 'Viết câu trả lời cuối bằng tiếng Việt', kind: 'Phản xạ', handles: ['COMPOSE_*'] },
]
const STAGE_VI = { core: 'Điều phối', intake: 'Tiếp nhận', predict: 'Dự đoán', knowledge: 'Tri thức', council: 'Hội đồng', rl: 'Can thiệp', reply: 'Trả lời' }
const PERF = [['REQUEST', 'p-req', 'yêu cầu'], ['INFORM', 'p-inf', 'trả kết quả'], ['CHALLENGE', 'p-ch', 'chất vấn'], ['RESPONSE', 'p-rs', 'trả lời chất vấn'], ['FAILURE', 'p-fail', 'lỗi']]
const pct = v => (v == null ? '–' : (v * 100).toFixed(1) + ' %')

// Tab "Hệ tác tử": bản vẽ hội đồng, vai trò từng tác tử, giao thức FIPA-ACL và chất lượng mô hình dự đoán.
function AgentsView({ systemInfo }) {
  const live = new Set(systemInfo?.agents || [])
  return (
    <div className="nb-page nb-agents">
      <section className="rail-card nb-page-head">
        <div className="nb-crumb">04. Hệ tác tử / MessageBus · FIPA-ACL</div>
        <h2>{systemInfo ? `${systemInfo.agents.length} tác tử` : 'Hệ tác tử'} trao đổi qua MessageBus</h2>
        <p className="rail-muted">Các tác tử không gọi hàm của nhau: mọi tương tác là một AgentMessage có performative, được bus định tuyến và ghi lại theo conversation_id.</p>
        <div className="nb-perf-legend">
          {PERF.map(([p, cls, vi]) => <span key={p}><b className={'nb-perf ' + cls}>{p}</b>{vi}</span>)}
        </div>
      </section>

      <ul className="nb-agent-cards">
        {AGENTS.map((ag, i) => {
          const Icon = ag.icon
          return (
            <li key={ag.name} className={`nb-agent-card stage-${ag.stage}${systemInfo && !live.has(ag.name) ? ' is-off' : ''}`}>
              <div className="nb-agent-head">
                <span className="nb-agent-icon"><Icon size={20} aria-hidden="true" /></span>
                <div className="nb-agent-title"><span className="nb-agent-stage">{String(i + 1).padStart(2, '0')} · {STAGE_VI[ag.stage]}</span><strong>{ag.name}</strong></div>
              </div>
              <p>{ag.role}</p>
              <div className="nb-agent-meta"><span className="nb-pill">{ag.kind}</span><code>src/agents/{ag.file}</code></div>
              <div className="nb-handles">{ag.handles.map(h => <span key={h}>{h}</span>)}</div>
            </li>
          )
        })}
      </ul>

      <div className="nb-agents-metrics">
        <section className="rail-card">
          <h2><Gauge size={17} aria-hidden="true" /> Các mô hình đã huấn luyện ({systemInfo?.model_registry?.length ?? '…'})</h2>
          <p className="rail-muted">{systemInfo?.prediction_dataset || '…'}</p>
          <table className="nb-metrics">
            <thead><tr><th scope="col">Mô hình</th><th scope="col">Loại</th><th scope="col">Dữ liệu</th><th scope="col">ROC-AUC</th><th scope="col">PR-AUC</th><th scope="col">Brier</th><th scope="col">Dùng trong hệ tác tử</th></tr></thead>
            <tbody>
              {(systemInfo?.model_registry || []).map(m => (
                <tr key={m.file} className={m.used_by ? undefined : 'is-lab'}>
                  <th scope="row">{m.name}<small>{m.file}{!m.exists && ' · thiếu file'}</small></th>
                  <td>{m.kind}</td><td>{m.dataset}</td>
                  <td>{pct(m.roc_auc)}</td><td>{pct(m.pr_auc)}</td><td>{m.brier ?? '–'}</td>
                  <td>{m.used_by ? <span className="nb-pill is-used">{m.used_by}</span> : <span className="nb-pill">chỉ thử nghiệm (notebook)</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="rail-muted">Số liệu NHANES lấy từ notebook/model/evaluation_leaderboard.csv (tập kiểm thử). LSTM, CNN và Q-learning không đánh giá bằng ROC-AUC nên để trống.</p>
          <p className="nb-llm"><Cpu size={14} aria-hidden="true" /> LLM: {systemInfo ? (systemInfo.llm.enabled ? `${systemInfo.llm.provider} · ${systemInfo.llm.model}` : 'tắt (chế độ luật)') : '…'}</p>
          <p className="rail-muted">Embedding: {systemInfo?.embedding || '…'} · KG: {systemInfo?.knowledge_graph || '…'}</p>
        </section>
      </div>

    </div>
  )
}

export default AgentsView
