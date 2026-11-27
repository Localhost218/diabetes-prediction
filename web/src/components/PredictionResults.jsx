import { useState } from 'react'
import { Lightbulb, Pill, ClipboardList, LoaderCircle, BookOpen, Pencil, Siren, CalendarCheck, MessageCircleQuestion, ChevronDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LEVELS = [
  { key: 'low', label: 'Thấp' },
  { key: 'moderate', label: 'Trung bình' },
  { key: 'high', label: 'Cao' },
  { key: 'very_high', label: 'Rất cao' },
]

const GLYCEMIC_STATUS = {
  normal: 'Đường huyết của bạn đang trong ngưỡng bình thường.',
  prediabetes_range: 'Đường huyết của bạn đang ở ngưỡng tiền đái tháo đường.',
  diabetes_range: 'Đường huyết của bạn đang ở ngưỡng đái tháo đường.',
  unknown: 'Chưa có xét nghiệm đường huyết nên đây chỉ là ước tính dựa trên yếu tố nguy cơ.',
}

// Gợi ý chung theo mức nguy cơ, theo khuyến cáo tầm soát của ADA/Bộ Y tế. Kế hoạch cá thể hóa lấy từ hội đồng.
const NEXT_STEPS = {
  low: ['Duy trì cân nặng hợp lý và ăn uống cân bằng.', 'Vận động ít nhất 150 phút mỗi tuần.', 'Kiểm tra đường huyết định kỳ khoảng 3 năm một lần, hoặc sớm hơn nếu có triệu chứng.'],
  moderate: ['Kiểm tra lại đường huyết lúc đói hoặc HbA1c trong vòng 12 tháng.', 'Nếu thừa cân, giảm 5–7% cân nặng giúp giảm rõ rệt nguy cơ.', 'Vận động 150 phút/tuần, hạn chế đồ ngọt và tinh bột trắng.'],
  high: ['Đặt lịch khám bác sĩ nội tiết hoặc nội tổng quát trong vài tuần tới.', 'Mang theo kết quả này và phiếu xét nghiệm khi đi khám.', 'Bắt đầu điều chỉnh ăn uống và vận động ngay từ bây giờ.'],
  very_high: ['Nên đi khám bác sĩ sớm, trong 1–2 tuần tới, để làm xét nghiệm xác nhận.', 'Mang theo kết quả này và phiếu xét nghiệm khi đi khám.', 'Không tự ý dùng thuốc trước khi có chỉ định của bác sĩ.'],
}

const FOLLOW_UPS = ['Giải thích kết quả này giúp tôi bằng lời dễ hiểu', 'Tôi nên ăn uống như thế nào?', 'Bài tập nào phù hợp với tôi?']

function RiskMeter({ level }) {
  const active = LEVELS.findIndex(l => l.key === level)
  return (
    <div className="risk-meter" role="img" aria-label={`Mức nguy cơ: ${LEVELS[active]?.label || 'chưa rõ'}`}>
      {LEVELS.map((l, i) => (
        <div key={l.key} className={`meter-seg seg-${l.key}${i === active ? ' is-active' : ''}${i < active ? ' is-passed' : ''}`}>
          <span className="meter-bar" />
          <span className="meter-label">{l.label}</span>
        </div>
      ))}
    </div>
  )
}

function Collapsible({ icon, title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className={`result-card collapsible${open ? ' is-open' : ''}`}>
      <button type="button" className="collapsible-head" aria-expanded={open} onClick={() => setOpen(o => !o)}>
        {icon}<h3>{title}</h3><ChevronDown size={18} className="chev" aria-hidden="true" />
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </section>
  )
}

function PredictionResults({ assessmentMessage, latestReply, isSending, onEdit, onAsk }) {
  const data = assessmentMessage?.data
  const consensus = data?.consensus
  const treatment = data?.treatment
  const clarification = latestReply?.data?.intent === 'RISK_ASSESSMENT' && !latestReply.data.consensus && !latestReply.is_error

  const header = (
    <div className="results-heading">
      <h2 id="results-heading">Kết quả của bạn</h2>
      {!isSending && <button type="button" className="btn-ghost" onClick={() => onEdit(0)}><Pencil size={15} aria-hidden="true" /> Sửa chỉ số</button>}
    </div>
  )

  if (isSending) {
    return (
      <div aria-busy="true">
        {header}
        <div className="results-loading" role="status">
          <LoaderCircle size={28} className="spin" aria-hidden="true" />
          <div>
            <strong>Đang phân tích chỉ số của bạn…</strong>
            <p>Các chuyên gia AI đang đối chiếu với hướng dẫn y khoa. Thường mất vài giây; bạn có thể theo dõi chi tiết ở khung trò chuyện.</p>
          </div>
        </div>
      </div>
    )
  }

  if (!consensus) {
    return (
      <div>
        {header}
        {latestReply?.is_error && <p className="prediction-error" role="alert">{latestReply.text}</p>}
        {clarification ? (
          <div className="result-notice" role="status">
            <h3>Cần thêm một chút thông tin</h3>
            <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{latestReply.text}</ReactMarkdown></div>
            <button type="button" className="btn-primary" onClick={() => onEdit(1)}>Bổ sung chỉ số</button>
          </div>
        ) : !latestReply?.is_error && <p className="result-placeholder">Chưa có kết quả. Hãy điền chỉ số để bắt đầu.</p>}
      </div>
    )
  }

  const level = consensus.risk_level
  return (
    <div>
      {header}

      <section className={`risk-hero level-${level.replaceAll('_', '-')}`}>
        <p className="risk-eyebrow">Mức nguy cơ đái tháo đường</p>
        <p className="risk-level-text">{LEVELS.find(l => l.key === level)?.label || consensus.risk_label}</p>
        <RiskMeter level={level} />
        <p className="risk-explain">{consensus.risk_label}</p>
        <p className="risk-score-line">Điểm nguy cơ <strong>{(consensus.consensus_risk_score * 100).toFixed(0)}%</strong>
          {typeof data.ml?.data_completeness === 'number' && <> · dữ liệu đầy đủ {(data.ml.data_completeness * 100).toFixed(0)}%</>}</p>
        {typeof data.ml?.risk_probability === 'number' && (
          <p className="risk-score-line">Mô hình yếu tố nguy cơ <strong>{(data.ml.risk_probability * 100).toFixed(0)}%</strong> (không dùng xét nghiệm;
            điểm nguy cơ ở trên đã kết hợp thêm ngưỡng xét nghiệm ADA)</p>
        )}
        <p className="glycemic-status">{GLYCEMIC_STATUS[consensus.glycemic_status] || consensus.glycemic_status}</p>
        <p className="relative-note">{consensus.relative_note || 'Kết quả chỉ mang tính tương đối, không thay thế chẩn đoán của bác sĩ.'}</p>
        {consensus.confirmation_required && <p className="confirmation-note">Một lần xét nghiệm chưa đủ để chẩn đoán. Bạn cần xét nghiệm lại để xác nhận.</p>}
        {consensus.safety_guardrails?.map((item, i) => (
          <p key={i} className={`evidence guardrail guardrail-${item.level}`}>{item.message}</p>
        ))}
      </section>

      <section className="result-card next-steps">
        <h3><CalendarCheck size={18} aria-hidden="true" /> Bạn nên làm gì tiếp theo</h3>
        <ol>
          {treatment?.recommended_action && <li><strong>{treatment.recommended_action}</strong></li>}
          {consensus.glycemic_status === 'unknown' && <li><strong>Làm xét nghiệm đường huyết lúc đói hoặc HbA1c</strong> (có ở hầu hết bệnh viện, phòng khám; nhịn ăn 8 giờ trước khi lấy máu) để biết chính xác.</li>}
          {(NEXT_STEPS[level] || []).map((s, i) => <li key={i}>{s}</li>)}
        </ol>
      </section>

      <Collapsible icon={<Lightbulb size={18} aria-hidden="true" />} title="Vì sao có kết quả này?" defaultOpen>
        <ul className="result-evidence">
          {consensus.positive_evidence_plus?.map((item, i) => <li key={`plus-${i}`} className="evidence plus">{item.replace(/^\[\+\]\s*/, '')}</li>)}
          {consensus.negative_contradictions_minus?.map((item, i) => <li key={`minus-${i}`} className="evidence minus">{item.replace(/^\[-\]\s*/, '')}</li>)}
        </ul>
        {data.ml?.top_risk_factors?.length > 0 && (
          <>
            <h4>Yếu tố ảnh hưởng nhiều nhất</h4>
            <ul className="risk-factor-list">
              {data.ml.top_risk_factors.map((factor, i) => (
                <li key={i}>
                  <span>{factor.feature_vi}</span>
                  <span className={factor.impact > 0 ? 'impact-up' : 'impact-down'}>{factor.impact > 0 ? '↑ làm tăng' : '↓ làm giảm'} nguy cơ</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </Collapsible>

      {treatment && (
        <Collapsible icon={<ClipboardList size={18} aria-hidden="true" />} title="Kế hoạch can thiệp gợi ý">
          {treatment.intervention_details && <p>{treatment.intervention_details}</p>}
          {treatment.monitoring_plan && <><h4>Theo dõi & tái khám</h4><p>{treatment.monitoring_plan}</p></>}
          {treatment.reference_drugs_vn?.length > 0 && (
            <>
              <h4><Pill size={16} aria-hidden="true" /> Thuốc tham khảo (chỉ để trao đổi với bác sĩ)</h4>
              <ul className="medication-list">{treatment.reference_drugs_vn.map((drug, i) => <li key={i}>{drug}</li>)}</ul>
            </>
          )}
          {treatment.notes?.length > 0 && (
            <><h4>Lưu ý riêng cho bạn</h4><ul className="treatment-notes">{treatment.notes.map((note, i) => <li key={i}>{note}</li>)}</ul></>
          )}
        </Collapsible>
      )}

      <section className="result-card urgent">
        <h3><Siren size={18} aria-hidden="true" /> Đi khám ngay nếu bạn có</h3>
        <p>Khát nước và đi tiểu nhiều bất thường, sụt cân không rõ lý do, mờ mắt, mệt lả, buồn nôn hoặc nôn, thở nhanh, lơ mơ.</p>
      </section>

      <section className="follow-ups">
        <h3><MessageCircleQuestion size={18} aria-hidden="true" /> Hỏi thêm trợ lý</h3>
        <div className="chip-row">
          {FOLLOW_UPS.map(q => <button type="button" key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>)}
        </div>
      </section>

      {data?.citations?.length > 0 && (
        <details className="result-sources">
          <summary><BookOpen size={15} aria-hidden="true" /> Nguồn tham khảo ({data.citations.length})</summary>
          <ul>{data.citations.map((citation, i) => <li key={i}>{citation.title}{citation.section && ` — ${citation.section}`}</li>)}</ul>
        </details>
      )}
    </div>
  )
}

export default PredictionResults
