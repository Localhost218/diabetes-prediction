import assert from 'node:assert/strict'
import test from 'node:test'
import { DIET_FIELDS, getBMI, profileToFormValues, validatePredictionForm, buildPredictionPrompt, buildProfileFields } from '../src/services/predictionForm.js'

const patient = { age_years: '52', bmi: '26.4', hba1c_percent: '6.1' }

test('calculates BMI from measurements and supports directly entered BMI', () => {
  assert.equal(getBMI({ height_cm: '165', weight_kg: '72' }), '26.4')
  assert.equal(getBMI({ bmi: '27', height_cm: '165' }), '27')
  assert.equal(getBMI({ height_cm: '0', weight_kg: '72' }), '')
  assert.equal(getBMI({}), '')
})

test('preserves a BMI updated through chat when earlier measurements are still recorded', () => {
  assert.equal(getBMI({ bmi: '29', height_cm: '165', weight_kg: '72' }), '29')
})

test('restores patient values without turning unknown history into negative answers', () => {
  const values = profileToFormValues([
    'Tuổi: 52 tuổi (bạn cung cấp)',
    'Giới tính: Nữ (bạn cung cấp)',
    'HbA1c: 6.1 % (từ file xét nghiệm.pdf)',
    'Tiền sử gia đình mắc ĐTĐ: Không (bạn cung cấp)',
    'Vận động thể lực thường xuyên: Có (bạn cung cấp)',
  ])
  assert.equal(values.age_years, '52')
  assert.equal(values.sex, '0')
  assert.equal(values.hba1c_percent, '6.1')
  assert.equal(values.family_history, 'no')
  assert.equal(values.phys_activity, 'yes')
  assert.equal(values.high_bp, '')
  assert.equal(values.weight_kg, '')
})

test('requires age, BMI, and at least one glycemic measurement', () => {
  assert.match(validatePredictionForm({}), /tuổi/i)
  assert.match(validatePredictionForm({ age_years: '52' }), /BMI/)
  assert.equal(validatePredictionForm({ age_years: '52', bmi: '26.4' }), '') // chưa xét nghiệm vẫn dự đoán (ước tính tương đối)
  assert.equal(validatePredictionForm(patient), '')
  assert.equal(validatePredictionForm({ age_years: '52', height_cm: '165', weight_kg: '72', glucose_fasting_mgdl: '129' }), '')
})

test('rejects implausible computed BMI and incomplete blood pressure measurements', () => {
  assert.match(validatePredictionForm({ ...patient, bmi: '', height_cm: '80', weight_kg: '300' }), /BMI/)
  assert.match(validatePredictionForm({ ...patient, systolic_bp: '130' }), /huyết áp/i)
  assert.match(validatePredictionForm({ ...patient, egfr: '250' }), /eGFR/)
  assert.match(validatePredictionForm({ ...patient, age_years: '52.5' }), /Tuổi/)
})

test('sends explicit units and only the information the user supplied', () => {
  const prompt = buildPredictionPrompt({ age_years: '52', bmi: '26.4', glucose_fasting_mgdl: '129' })
  assert.match(prompt, /52 tuổi/)
  assert.match(prompt, /BMI: 26\.4 kg\/m²/)
  assert.match(prompt, /Đường huyết lúc đói: 129 mg\/dL/)
  assert.doesNotMatch(prompt, /HbA1c:|Giới tính:|gia đình|hút thuốc|tăng huyết áp/)
})

test('formats blood pressure and distinguishes positive and negative medical history', () => {
  const prompt = buildPredictionPrompt({
    ...patient, sex: '0', systolic_bp: '130', diastolic_bp: '80',
    family_history: 'no', high_bp: 'yes', smoker: 'no', phys_activity: 'yes',
  })
  assert.match(prompt, /Giới tính: nữ/)
  assert.match(prompt, /Huyết áp: 130\/80 mmHg/)
  assert.match(prompt, /Gia đình không ai mắc đái tháo đường/)
  assert.match(prompt, /Tôi bị tăng huyết áp/)
  assert.match(prompt, /Tôi không hút thuốc/)
  assert.match(prompt, /Tôi tập thể dục thường xuyên/)
})

test('keeps affirmative history ahead of negations so text intake preserves both answers', () => {
  const prompt = buildPredictionPrompt({ ...patient, smoker: 'no', phys_activity: 'yes', pregnant: 'yes' })
  assert.ok(prompt.indexOf('Tôi tập thể dục thường xuyên.') < prompt.indexOf('Tôi không hút thuốc.'))
  assert.ok(prompt.indexOf('Tôi đang mang thai.') < prompt.indexOf('Tôi không hút thuốc.'))
})


test('builds profile fields for saving without labs: skips blanks and lets backend derive BMI', () => {
  assert.deepEqual(buildProfileFields({ age_years: '52', sex: '1', height_cm: '165', weight_kg: '72', bmi: '26.4', waist_cm: '', family_history: 'yes', smoker: 'no' }),
    { age_years: '52', height_cm: '165', weight_kg: '72', family_history: 'yes', smoker: 'no', sex: '1' })
  assert.deepEqual(buildProfileFields({ bmi: '27' }), { bmi: '27' })
  assert.deepEqual(buildProfileFields({}), {})
})

test('lifestyle answers round-trip through the profile summary, the prompt and the saved fields', () => {
  const values = profileToFormValues(['Ăn rau xanh mỗi ngày: Không (nhập form)', 'Ăn trái cây mỗi ngày: Có (bạn cung cấp)', 'Giờ ngủ mỗi đêm: 5 giờ (nhập form)'])
  assert.equal(values.vegetables_daily, 'no')
  assert.equal(values.fruits_daily, 'yes')
  assert.equal(values.sleep_hours, '5')
  const prompt = buildPredictionPrompt({ age_years: '40', bmi: '24', ...values })
  assert.match(prompt, /Giờ ngủ mỗi đêm: 5 giờ/)
  assert.match(prompt, /ít ăn rau/)
  assert.deepEqual(buildProfileFields(values), { sleep_hours: '5', vegetables_daily: 'no', fruits_daily: 'yes' })
  assert.equal(DIET_FIELDS.length, 2)
})
