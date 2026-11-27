// Labels, units, and ranges follow the backend PatientProfile contract.
export const NUMBER_FIELDS = [
  { name: 'age_years', label: 'Tuổi', unit: 'tuổi', min: 1, max: 120, step: 1, placeholder: '52' },
  { name: 'height_cm', label: 'Chiều cao', unit: 'cm', min: 80, max: 230, placeholder: '165' },
  { name: 'weight_kg', label: 'Cân nặng', unit: 'kg', min: 20, max: 300, placeholder: '72' },
  { name: 'bmi', label: 'BMI', unit: 'kg/m²', min: 10, max: 70, placeholder: '26.4', info: 'Chỉ số khối cơ thể = cân nặng ÷ (chiều cao)². Với người châu Á: 18,5–22,9 là bình thường, từ 23 là thừa cân.' },
  { name: 'waist_cm', label: 'Vòng eo', unit: 'cm', min: 40, max: 200, placeholder: '90', info: 'Đo ngang rốn, lúc thở ra nhẹ. Nam ≥ 90 cm, nữ ≥ 80 cm là béo bụng.' },
  { name: 'glucose_fasting_mgdl', label: 'Đường huyết lúc đói', unit: 'mg/dL', min: 30, max: 700, placeholder: '110', info: 'Đo sau nhịn ăn ít nhất 8 giờ. Bình thường < 5,6 mmol/L (100 mg/dL); 5,6–6,9 là tiền đái tháo đường; ≥ 7,0 cần khám.' },
  { name: 'hba1c_percent', label: 'HbA1c', unit: '%', min: 3, max: 20, placeholder: '6.1', info: 'Phản ánh đường huyết trung bình 2–3 tháng gần nhất. Bình thường < 5,7%; 5,7–6,4% là tiền đái tháo đường; ≥ 6,5% cần khám.' },
  { name: 'systolic_bp', label: 'Huyết áp tâm thu', unit: 'mmHg', min: 60, max: 260, placeholder: '120', info: 'Số trên khi đo huyết áp. Bình thường dưới 120 mmHg.' },
  { name: 'diastolic_bp', label: 'Huyết áp tâm trương', unit: 'mmHg', min: 30, max: 160, placeholder: '80', info: 'Số dưới khi đo huyết áp. Bình thường dưới 80 mmHg.' },
  { name: 'total_cholesterol_mgdl', label: 'Cholesterol toàn phần', unit: 'mg/dL', min: 50, max: 600, placeholder: '190', info: 'Mỡ máu toàn phần. Mong muốn dưới 200 mg/dL (5,2 mmol/L).' },
  { name: 'sleep_hours', label: 'Giờ ngủ mỗi đêm', unit: 'giờ', min: 2, max: 14, placeholder: '7', info: 'Số giờ ngủ trung bình mỗi đêm. Ngủ dưới 6 giờ hoặc từ 9 giờ trở lên đều đi kèm nguy cơ đái tháo đường cao hơn.' },
  { name: 'egfr', label: 'eGFR', unit: 'mL/phút/1.73m²', min: 1, max: 200, placeholder: '90', info: 'Mức lọc cầu thận, cho biết thận hoạt động tốt thế nào. Từ 90 trở lên là bình thường.' },
]

export const HISTORY_FIELDS = [
  { name: 'family_history', label: 'Tiền sử gia đình mắc ĐTĐ', yes: 'Gia đình có người mắc đái tháo đường.', no: 'Gia đình không ai mắc đái tháo đường.' },
  { name: 'high_bp', label: 'Tăng huyết áp', yes: 'Tôi bị tăng huyết áp.', no: 'Tôi không bị tăng huyết áp.' },
  { name: 'smoker', label: 'Hút thuốc', yes: 'Tôi hút thuốc.', no: 'Tôi không hút thuốc.' },
  { name: 'phys_activity', label: 'Vận động thể lực thường xuyên', yes: 'Tôi tập thể dục thường xuyên.', no: 'Tôi ít vận động.' },
  { name: 'diagnosed_diabetes', label: 'Đã được chẩn đoán đái tháo đường', yes: 'Tôi đã được chẩn đoán đái tháo đường.', no: 'Tôi chưa được chẩn đoán đái tháo đường.' },
  { name: 'glucose_lowering_meds', label: 'Đang dùng thuốc hạ đường huyết / insulin', yes: 'Tôi đang dùng thuốc hạ đường huyết.', no: 'Tôi chưa dùng thuốc hạ đường huyết.' },
  { name: 'pregnant', label: 'Đang mang thai', yes: 'Tôi đang mang thai.', no: 'Tôi không mang thai.' },
]

// Rau, trái cây (đặc trưng mô hình BRFSS; khớp DIET_FIELDS ở backend); câu "yes"/"no" ghép vào tin nhắn gửi hội đồng
export const DIET_FIELDS = [
  { name: 'vegetables_daily', label: 'Ăn rau xanh mỗi ngày', hint: 'Có rau trong hầu hết các bữa', yes: 'Tôi ăn rau xanh mỗi ngày.', no: 'Tôi ít ăn rau.' },
  { name: 'fruits_daily', label: 'Ăn trái cây mỗi ngày', hint: 'Hầu hết các ngày trong tuần', yes: 'Tôi ăn trái cây mỗi ngày.', no: 'Tôi ít ăn trái cây.' },
]

export function getBMI(values) {
  if (values.bmi) return values.bmi
  const height = Number(values.height_cm), weight = Number(values.weight_kg)
  if (height > 0 && weight > 0) return (weight / (height / 100) ** 2).toFixed(1)
  return ''
}

export function profileToFormValues(summary = []) {
  const recorded = Object.fromEntries(summary.map(line => {
    const colon = line.indexOf(':')
    return [line.slice(0, colon), line.slice(colon + 1).trim()]
  }))
  const values = { sex: recorded['Giới tính']?.startsWith('Nam') ? '1' : recorded['Giới tính']?.startsWith('Nữ') ? '0' : '' }
  for (const field of NUMBER_FIELDS) values[field.name] = recorded[field.label]?.match(/^\d+(?:[.,]\d+)?/)?.[0].replace(',', '.') || ''
  for (const field of [...HISTORY_FIELDS, ...DIET_FIELDS]) {
    const answer = recorded[field.label]
    values[field.name] = answer?.startsWith('Có') ? 'yes' : answer?.startsWith('Không') ? 'no' : ''
  }
  return values
}

/** Kiểm tra khoảng giá trị của các trường đã điền (bỏ qua trường trống). `names` mặc định là tất cả. */
export function validateFieldRanges(values, names) {
  for (const field of NUMBER_FIELDS) {
    if (names && !names.includes(field.name)) continue
    const value = field.name === 'bmi' ? getBMI(values) : values[field.name]
    if (value === '' || value == null) continue
    const number = Number(value)
    if (!Number.isFinite(number) || number < field.min || number > field.max || (field.step === 1 && !Number.isInteger(number))) {
      return `${field.label} cần nằm trong khoảng ${field.min}–${field.max} ${field.unit}${field.step === 1 ? ' và là số nguyên' : ''}.`
    }
  }
  return ''
}

export function validatePredictionForm(values) {
  if (!values.age_years) return 'Vui lòng nhập tuổi.'
  if (!getBMI(values)) return 'Vui lòng nhập BMI hoặc cả chiều cao và cân nặng.'
  const rangeError = validateFieldRanges(values)
  if (rangeError) return rangeError
  if (!!values.systolic_bp !== !!values.diastolic_bp) return 'Vui lòng nhập cả huyết áp tâm thu và tâm trương.'
  return ''
}

export function buildPredictionPrompt(values) {
  const lines = ['Hãy dự đoán nguy cơ đái tháo đường của tôi từ các chỉ số sau:']
  for (const field of NUMBER_FIELDS) {
    if (field.name === 'systolic_bp' || field.name === 'diastolic_bp') continue
    const value = field.name === 'bmi' ? getBMI(values) : values[field.name]
    if (value !== '' && value != null) lines.push(`${field.label}: ${value} ${field.unit}.`)
  }
  if (values.sex !== '' && values.sex != null) lines.push(`Giới tính: ${values.sex === '1' ? 'nam' : 'nữ'}.`)
  if (values.systolic_bp && values.diastolic_bp) lines.push(`Huyết áp: ${values.systolic_bp}/${values.diastolic_bp} mmHg.`)
  // The text intake checks nearby negations; group affirmative facts first to keep answers independent.
  for (const answer of ['yes', 'no']) {
    for (const field of [...HISTORY_FIELDS, ...DIET_FIELDS]) {
      if (values[field.name] === answer) lines.push(field[answer])
    }
  }
  lines.push('Giải thích các yếu tố dẫn đến kết quả và đề xuất kế hoạch can thiệp, gồm thông tin thuốc tham khảo nếu phù hợp.')
  return lines.join('\n')
}

/** Các trường form gửi lên PUT /api/sessions/{id}/profile (bỏ ô trống; BMI để backend tự tính khi có chiều cao, cân nặng). */
export function buildProfileFields(values) {
  const names = [...NUMBER_FIELDS.map(f => f.name), ...HISTORY_FIELDS.map(f => f.name), ...DIET_FIELDS.map(f => f.name), 'sex']
  const fields = {}
  for (const name of names) {
    const v = values[name]
    if (v === '' || v == null) continue
    if (name === 'bmi' && values.height_cm && values.weight_kg) continue
    fields[name] = v
  }
  return fields
}
