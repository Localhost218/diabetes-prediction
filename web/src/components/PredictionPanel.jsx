import { useId, useMemo, useState } from 'react'
import { ArrowLeft, ArrowRight, Check, ChevronDown, Info, ShieldCheck, Stethoscope, User, FlaskConical, HeartPulse } from 'lucide-react'
import { NUMBER_FIELDS, HISTORY_FIELDS, DIET_FIELDS, getBMI, profileToFormValues, validatePredictionForm, validateFieldRanges, buildPredictionPrompt, buildProfileFields } from '../services/predictionForm'
import PredictionResults from './PredictionResults'
import NoLabsGuide from './NoLabsGuide'

const MGDL_PER_MMOL = 18.016
const FIELD = Object.fromEntries(NUMBER_FIELDS.map(f => [f.name, f]))
const STEPS = [
  { title: 'Về bạn', icon: User, fields: ['age_years', 'height_cm', 'weight_kg', 'bmi', 'waist_cm'] },
  { title: 'Xét nghiệm', icon: FlaskConical, fields: ['glucose_fasting_mgdl', 'hba1c_percent', 'systolic_bp', 'diastolic_bp', 'total_cholesterol_mgdl', 'egfr'] },
  { title: 'Lối sống & tiền sử', icon: HeartPulse, fields: ['sleep_hours'] },
]

// Giải thích thuật ngữ: ẩn mặc định để form gọn, bấm (i) để xem
function InfoButton({ text, open, onToggle, controls, label }) {
  if (!text) return null
  return (
    <button type="button" className={`info-btn${open ? ' is-open' : ''}`} onClick={onToggle}
      aria-expanded={open} aria-controls={controls} aria-label={`${label} là gì?`} title={`${label} là gì?`}>
      <Info size={15} aria-hidden="true" />
    </button>
  )
}

function NumberField({ field, value, onChange, required = false, readOnly = false, unit, unitSwitch, hint }) {
  const id = `patient-${field.name}`
  const infoId = useId()
  const [showInfo, setShowInfo] = useState(false)
  return (
    <div className="patient-field">
      <div className="patient-label-row">
        <label htmlFor={id}>{field.label}{required && <span className="required-mark"> *</span>}</label>
        <InfoButton text={field.info} open={showInfo} onToggle={() => setShowInfo(o => !o)} controls={infoId} label={field.label} />
        {unitSwitch}
      </div>
      <div className={`patient-input${readOnly ? ' is-calculated' : ''}`}>
        <input id={id} name={field.name} type="number" inputMode={field.step === 1 ? 'numeric' : 'decimal'}
          step="any" placeholder={field.placeholder && `VD: ${field.placeholder}`}
          value={value || ''} onChange={e => onChange(field.name, e.target.value)} readOnly={readOnly} />
        <span className="patient-unit">{unit || field.unit}</span>
      </div>
      {showInfo && <p className="field-info" id={infoId}>{field.info}</p>}
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  )
}

function Choice({ name, value, onChange, options, label }) {
  return (
    <div className="choice-group" role="radiogroup" aria-label={label}>
      {options.map(o => (
        <button type="button" key={o.value} role="radio" aria-checked={value === o.value}
          className={`choice${value === o.value ? ' is-selected' : ''}`} onClick={() => onChange(name, value === o.value ? '' : o.value)}>
          {value === o.value && <Check size={14} aria-hidden="true" />}{o.label}
        </button>
      ))}
    </div>
  )
}

const YES_NO = [{ value: 'yes', label: 'Có' }, { value: 'no', label: 'Không' }]

function PredictionPanel({ profileSummary, isSending, disabled, onPredict, onAsk, assessmentMessage, latestReply, initialView, onComplete }) {
  const recorded = useMemo(() => profileToFormValues(profileSummary), [profileSummary])
  const [edits, setEdits] = useState({})
  const [error, setError] = useState('')
  const [step, setStep] = useState(0)
  const [glucoseUnit, setGlucoseUnit] = useState('mmol')
  const [glucoseMmol, setGlucoseMmol] = useState('')
  const [hasLabs, setHasLabs] = useState(() => (recorded.glucose_fasting_mgdl || recorded.hba1c_percent ? 'yes' : ''))
  const noLabs = hasLabs === 'no'
  const values = { ...recorded, ...edits }
  const measuredBMI = getBMI({ ...values, bmi: '' })
  const calculatedBMI = !!measuredBMI && (!values.bmi || Number(values.bmi) === Number(measuredBMI))

  // Khi hội đồng trả về một kết quả mới (từ form hoặc từ chat) thì chuyển sang màn kết quả
  const assessId = assessmentMessage?.id
  const [view, setView] = useState(initialView || (assessId ? 'result' : 'form'))
  const [seenAssess, setSeenAssess] = useState(assessId)
  if (assessId !== seenAssess) {
    setSeenAssess(assessId)
    if (assessId) setView('result')
  }

  const updateField = (name, value) => {
    setEdits(prev => ({ ...prev, [name]: value, ...(['height_cm', 'weight_kg'].includes(name) ? { bmi: '' } : {}) }))
    setError('')
  }

  const mgdl = values.glucose_fasting_mgdl
  const glucoseShown = glucoseUnit === 'mgdl' ? mgdl
    : (glucoseMmol !== '' && Math.round(Number(glucoseMmol) * MGDL_PER_MMOL) === Number(mgdl) ? glucoseMmol
      : mgdl ? (Number(mgdl) / MGDL_PER_MMOL).toFixed(1) : '')
  const setGlucose = (_, v) => {
    if (glucoseUnit === 'mgdl') return updateField('glucose_fasting_mgdl', v)
    setGlucoseMmol(v)
    updateField('glucose_fasting_mgdl', v === '' ? '' : String(Math.round(Number(v) * MGDL_PER_MMOL)))
  }

  const stepError = (i) => {
    if (i === 0) {
      if (!values.age_years) return 'Vui lòng nhập tuổi của bạn.'
      if (!getBMI(values)) return 'Vui lòng nhập chiều cao và cân nặng (hoặc BMI nếu bạn đã biết).'
    }
    if (i === 1) {
      if (!hasLabs) return 'Vui lòng cho biết bạn đã có kết quả xét nghiệm đường huyết chưa.'
      if (noLabs) return validateFieldRanges(values, ['systolic_bp', 'diastolic_bp', 'total_cholesterol_mgdl', 'egfr'])
      if (!values.glucose_fasting_mgdl && !values.hba1c_percent) return 'Cần ít nhất một chỉ số: đường huyết lúc đói hoặc HbA1c.'
      if (!!values.systolic_bp !== !!values.diastolic_bp) return 'Vui lòng nhập cả hai số huyết áp (trên và dưới).'
    }
    return validateFieldRanges(values, STEPS[i].fields)
  }

  const next = () => {
    const err = stepError(step)
    if (err) { setError(err); return }
    setError('')
    setStep(s => s + 1)
  }

  const submit = async (event) => {
    event.preventDefault()
    if (step < STEPS.length - 1) { next(); return }
    if (isSending || disabled) return
    const validationError = validatePredictionForm(values)
    if (validationError) { setError(validationError); return }
    setError('')
    setView('result')
    try {
      const result = await onPredict(buildPredictionPrompt(values))
      if (result?.bot_message?.data?.profile_summary) setEdits({})
    } catch (err) {
      setError(err.message || 'Không thể hoàn tất dự đoán. Vui lòng thử lại.')
      setView('form')
    }
  }

  const editForm = (toStep = 0) => { setStep(toStep); setView('form') }

  // "Hoàn thành": lưu những gì đã nhập vào hồ sơ phiên, không cần xét nghiệm, để trò chuyện tiếp với hồ sơ đó
  const [saving, setSaving] = useState(false)
  const complete = async () => {
    const fields = buildProfileFields(values)
    if (!Object.keys(fields).length) { setError('Bạn chưa nhập chỉ số nào để lưu.'); return }
    const rangeError = validateFieldRanges(values)
    if (rangeError) { setError(rangeError); return }
    setError(''); setSaving(true)
    try {
      await onComplete(fields)
    } catch (err) {
      setError(err.message || 'Không lưu được hồ sơ. Vui lòng thử lại.')
      setSaving(false)
    }
  }
  const completeButton = onComplete && (
    <button type="button" className="btn-secondary nb-complete-btn" onClick={complete} disabled={saving || isSending || disabled}>
      <Check size={18} aria-hidden="true" /> {saving ? 'Đang lưu…' : 'Hoàn thành'}
    </button>
  )

  if (view === 'guide') {
    return (
      <section className="prediction-panel" aria-labelledby="guide-heading">
        <NoLabsGuide values={values} onAsk={onAsk}
          onHaveResults={() => { setHasLabs('yes'); editForm(1) }} onEdit={() => editForm(0)} />
        {error && <p className="prediction-error" role="alert">{error}</p>}
        {completeButton && <div className="nb-guide-complete">{completeButton}<p className="rail-muted">Lưu hồ sơ hiện có để hỏi trợ lý; khi có xét nghiệm, bổ sung để hội đồng đánh giá.</p></div>}
      </section>
    )
  }

  if (view === 'result') {
    return (
      <section className="prediction-panel" aria-labelledby="results-heading">
        <PredictionResults assessmentMessage={assessmentMessage} latestReply={latestReply} isSending={isSending}
          onEdit={editForm} onAsk={onAsk} />
      </section>
    )
  }

  const nf = (name, options = {}) => (
    <NumberField key={name} field={FIELD[name]} value={name === 'bmi' ? getBMI(values) : values[name]} onChange={updateField} {...options} />
  )

  return (
    <section className="prediction-panel" aria-labelledby="patient-heading">
      {step === 0 && <div className="intake-intro">
        <span className="intro-icon"><Stethoscope size={22} aria-hidden="true" /></span>
        <div>
          <h2 id="patient-heading">Kiểm tra nguy cơ đái tháo đường</h2>
          <p>Khoảng 2 phút · 3 bước ngắn · không cần đăng nhập</p>
        </div>
      </div>}

      <ol className="stepper" aria-label="Các bước">
        {STEPS.map((s, i) => {
          const Icon = s.icon
          const state = i < step ? 'done' : i === step ? 'current' : 'todo'
          return (
            <li key={s.title} className={`step is-${state}`} aria-current={state === 'current' ? 'step' : undefined}>
              <button type="button" disabled={i > step || isSending} onClick={() => { setError(''); setStep(i) }}>
                <span className="step-dot">{state === 'done' ? <Check size={14} /> : <Icon size={14} />}</span>
                <span className="step-title">{s.title}</span>
              </button>
            </li>
          )
        })}
      </ol>

      <form className="patient-form" onSubmit={submit} noValidate>
        <fieldset disabled={isSending || disabled}>
          {step === 0 && (
            <>
              <legend className="step-question">Cho chúng tôi biết đôi chút về bạn</legend>
              <p className="step-help">Mục có dấu <span className="required-mark">*</span> là bắt buộc, còn lại chưa rõ cứ để trống.</p>
              <div className="patient-fields-grid">
                {nf('age_years', { required: true })}
                <div className="patient-field">
                  <span className="patient-label">Giới tính</span>
                  <Choice name="sex" value={values.sex} onChange={updateField} label="Giới tính"
                    options={[{ value: '1', label: 'Nam' }, { value: '0', label: 'Nữ' }]} />
                </div>
                {nf('height_cm', { required: !values.bmi })}
                {nf('weight_kg', { required: !values.bmi })}
                {nf('bmi', { readOnly: calculatedBMI, hint: calculatedBMI ? 'Đã tự tính từ chiều cao, cân nặng.' : 'Tự tính từ chiều cao, cân nặng.' })}
                {nf('waist_cm')}
              </div>
            </>
          )}

          {step === 1 && (
            <>
              <legend className="step-question">Bạn đã có kết quả xét nghiệm đường huyết chưa?</legend>
              <p className="step-help">Ví dụ phiếu khám sức khỏe định kỳ có ghi "Glucose" hoặc "HbA1c".</p>
              <Choice name="hasLabs" value={hasLabs} onChange={(_, v) => { setError(''); setHasLabs(v) }} label="Đã có kết quả xét nghiệm"
                options={[{ value: 'yes', label: 'Có, tôi có kết quả' }, { value: 'no', label: 'Chưa / không nhớ' }]} />

              {noLabs && (
                <div className="no-labs-note">
                  <strong>Không sao cả.</strong> Hội đồng vẫn ước tính được nguy cơ từ tuổi, cân nặng, tiền sử và thói quen ăn uống,
                  nhưng kết quả <strong>chỉ mang tính tương đối</strong>. Đo đường huyết lúc đói hoặc HbA1c sẽ cho kết quả chính xác hơn.
                  <button type="button" className="link-btn" onClick={() => setView('guide')}>Nên đi xét nghiệm gì, ở đâu?</button>
                </div>
              )}

              {hasLabs === 'yes' && (
                <>
                  <p className="step-help step-help-tight">Chỉ cần <strong>một trong hai</strong> chỉ số. Có phiếu giấy? Gửi ảnh ở khung trò chuyện, trợ lý sẽ đọc giúp.</p>
                  <div className="patient-fields-grid">
                    <NumberField field={FIELD.glucose_fasting_mgdl} value={glucoseShown} onChange={setGlucose}
                      required={!values.hba1c_percent} unit={glucoseUnit === 'mmol' ? 'mmol/L' : 'mg/dL'}
                      unitSwitch={
                        <div className="unit-switch" role="group" aria-label="Đơn vị đường huyết">
                          {[['mmol', 'mmol/L'], ['mgdl', 'mg/dL']].map(([u, l]) => (
                            <button type="button" key={u} aria-pressed={glucoseUnit === u} onClick={() => { setGlucoseMmol(''); setGlucoseUnit(u) }}>{l}</button>
                          ))}
                        </div>
                      } />
                    {nf('hba1c_percent', { required: !values.glucose_fasting_mgdl })}
                  </div>
                </>
              )}

              {hasLabs && (
                <details className="more-fields" open={['systolic_bp', 'diastolic_bp', 'total_cholesterol_mgdl', 'egfr'].some(n => values[n]) || undefined}>
                  <summary>Thêm chỉ số khác nếu biết <span>(huyết áp, mỡ máu, chức năng thận)</span><ChevronDown size={16} aria-hidden="true" /></summary>
                  <div className="patient-fields-grid">
                    {nf('systolic_bp', { required: !!values.diastolic_bp })}
                    {nf('diastolic_bp', { required: !!values.systolic_bp })}
                    {nf('total_cholesterol_mgdl')}
                    {nf('egfr')}
                  </div>
                </details>
              )}
            </>
          )}

          {step === 2 && (
            <>
              <legend className="step-question">Lối sống & tiền sử sức khỏe</legend>
              <p className="step-help">Bỏ qua câu bạn không chắc. Bấm lại để bỏ chọn.{noLabs && ' Các câu này giúp chỉ ra yếu tố nguy cơ của bạn.'}</p>
              <ul className="history-list">
                {HISTORY_FIELDS.filter(f => f.name !== 'pregnant' || values.sex !== '1').map(field => (
                  <li key={field.name}>
                    <span>{field.label}</span>
                    <Choice name={field.name} value={values[field.name]} onChange={updateField} label={field.label} options={YES_NO} />
                  </li>
                ))}
              </ul>
              <p className="subsection-title">Ăn uống & giấc ngủ</p>
              <div className="patient-fields-grid">{nf('sleep_hours')}</div>
              <ul className="history-list diet-list">
                {DIET_FIELDS.map(field => (
                  <li key={field.name}>
                    <span>{field.label}<small>{field.hint}</small></span>
                    <Choice name={field.name} value={values[field.name]} onChange={updateField} label={field.label} options={YES_NO} />
                  </li>
                ))}
              </ul>
            </>
          )}
        </fieldset>

        <div className="form-footer">
        {error && <p className="prediction-error" role="alert">{error}</p>}
        <div className="form-nav">
          {step > 0 && (
            <button type="button" className="btn-secondary" onClick={() => { setError(''); setStep(s => s - 1) }} disabled={isSending}>
              <ArrowLeft size={18} aria-hidden="true" /> Quay lại
            </button>
          )}
          {completeButton}
          <button className="btn-primary" type="submit" disabled={isSending || disabled}>
            {step < STEPS.length - 1
              ? <>Tiếp tục <ArrowRight size={18} aria-hidden="true" /></>
              : <>{noLabs ? 'Xem kết quả ước tính' : 'Xem kết quả'} <ArrowRight size={18} aria-hidden="true" /></>}
          </button>
        </div>
        <p className="privacy-note"><ShieldCheck size={14} aria-hidden="true" />Thông tin chỉ dùng cho lần tư vấn này, tự xóa khi bạn đóng trang.</p>
        {assessId && <button type="button" className="link-btn" onClick={() => setView('result')}>Xem lại kết quả gần nhất</button>}
        </div>
      </form>

    </section>
  )
}

export default PredictionPanel
