import { FlaskConical, ListChecks, MapPin, Pencil, Siren, MessageCircleQuestion, ClipboardCheck } from 'lucide-react'
import { getBMI } from '../services/predictionForm'

// Yếu tố nguy cơ theo tiêu chí tầm soát đã công bố (ADA Standards of Care; ngưỡng BMI/vòng eo cho người châu Á của WHO/IDF).
// Đây là danh sách đối chiếu, không phải dự đoán: hội đồng AI cần ít nhất một chỉ số đường huyết mới ước tính được.
function riskFactors(v) {
  const out = []
  const age = Number(v.age_years), bmi = Number(getBMI(v)), waist = Number(v.waist_cm)
  if (age >= 35) out.push(`Từ 35 tuổi trở lên (${age} tuổi): ADA khuyến cáo mọi người từ tuổi này nên xét nghiệm định kỳ.`)
  if (bmi >= 25) out.push(`BMI ${bmi}: béo phì theo chuẩn người châu Á (từ 25).`)
  else if (bmi >= 23) out.push(`BMI ${bmi}: thừa cân theo chuẩn người châu Á (từ 23).`)
  if (waist && ((v.sex === '1' && waist >= 90) || (v.sex === '0' && waist >= 80))) out.push(`Vòng eo ${waist} cm: béo bụng, liên quan chặt với kháng insulin.`)
  if (v.family_history === 'yes') out.push('Có người thân mắc đái tháo đường.')
  if (v.high_bp === 'yes' || (Number(v.systolic_bp) >= 130 || Number(v.diastolic_bp) >= 80)) out.push('Tăng huyết áp (từ 130/80 mmHg hoặc đang điều trị).')
  if (v.phys_activity === 'no') out.push('Ít vận động.')
  if (v.smoker === 'yes') out.push('Hút thuốc.')
  return out
}

const FOLLOW_UPS = ['Xét nghiệm đường huyết lúc đói cần chuẩn bị gì?', 'HbA1c khác gì đường huyết lúc đói?', 'Làm sao giảm nguy cơ tiểu đường?']

function NoLabsGuide({ values, onHaveResults, onEdit, onAsk }) {
  const factors = riskFactors(values)
  const diagnosed = values.diagnosed_diabetes === 'yes' || values.glucose_lowering_meds === 'yes'

  return (
    <div>
      <div className="results-heading">
        <h2 id="guide-heading">Hướng dẫn cho bạn</h2>
        <button type="button" className="btn-ghost" onClick={onEdit}><Pencil size={15} aria-hidden="true" /> Sửa thông tin</button>
      </div>

      <section className="guide-hero">
        <span className="guide-hero-icon"><FlaskConical size={26} aria-hidden="true" /></span>
        <div>
          <p className="guide-hero-title">Bạn nên làm một xét nghiệm đường huyết</p>
          <p>Chưa có chỉ số đường huyết thì chưa thể đánh giá chính xác nguy cơ. Tin tốt: xét nghiệm nhanh, rẻ và có ở hầu hết cơ sở y tế.</p>
        </div>
      </section>

      {diagnosed && (
        <p className="evidence guardrail">Bạn cho biết đã được chẩn đoán hoặc đang dùng thuốc: hãy tái khám và xét nghiệm theo lịch của bác sĩ điều trị.</p>
      )}

      <section className="result-card">
        <h3><ListChecks size={18} aria-hidden="true" /> Yếu tố nguy cơ bạn đang có</h3>
        {factors.length > 0 ? (
          <>
            <ul className="factor-checklist">{factors.map(f => <li key={f}>{f}</li>)}</ul>
            <p className="guide-muted">Càng nhiều yếu tố, càng nên xét nghiệm sớm. Đây là đối chiếu theo tiêu chí tầm soát, không phải chẩn đoán.</p>
          </>
        ) : (
          <p className="guide-muted">Chưa thấy yếu tố nguy cơ nổi bật từ thông tin bạn cung cấp. Bạn vẫn nên xét nghiệm định kỳ, đặc biệt từ 35 tuổi.</p>
        )}
      </section>

      <section className="result-card next-steps">
        <h3><ClipboardCheck size={18} aria-hidden="true" /> Chọn một trong hai xét nghiệm</h3>
        <div className="test-options">
          <div className="test-option">
            <strong>Đường huyết lúc đói</strong>
            <span>Nhịn ăn ít nhất 8 giờ (được uống nước lọc), lấy máu buổi sáng.</span>
          </div>
          <div className="test-option">
            <strong>HbA1c</strong>
            <span>Không cần nhịn ăn. Phản ánh đường huyết 2–3 tháng gần nhất.</span>
          </div>
        </div>
        <p className="guide-where"><MapPin size={15} aria-hidden="true" /> Bệnh viện, phòng khám đa khoa, trạm y tế hoặc phòng xét nghiệm tư nhân. Thường có kết quả trong ngày.</p>
      </section>

      <button type="button" className="btn-primary guide-cta" onClick={onHaveResults}>Tôi đã có kết quả, nhập chỉ số</button>

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
    </div>
  )
}

export default NoLabsGuide
