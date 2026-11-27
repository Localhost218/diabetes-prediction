import assert from 'node:assert/strict'
import test from 'node:test'
import { parseProfileLine, groupProfile, glycemicBand, planStatus, activeAgent } from '../src/services/workspace.js'

const summary = [
  'Tuổi: 52 tuổi (bạn cung cấp)',
  'BMI: 26.4 kg/m² (tính toán)',
  'Đường huyết lúc đói: 129.7 mg/dL (từ file phieu.jpg)',
  'HbA1c: 6.1 % (nhập form)',
  'Tiền sử gia đình mắc ĐTĐ: Có (bạn cung cấp)',
]

test('parses value, source tag and number from a profile line', () => {
  assert.deepEqual(parseProfileLine(summary[2]), { label: 'Đường huyết lúc đói', value: '129.7 mg/dL', source: 'từ file phieu.jpg', tag: 'file', num: 129.7 })
  assert.equal(parseProfileLine(summary[1]).tag, 'tính')
  assert.equal(parseProfileLine(summary[3]).tag, 'form')
  assert.equal(parseProfileLine(summary[4]).num, null)
})

test('groups profile into notebook sections and computes completeness', () => {
  const { groups, filled, total, percent } = groupProfile(summary)
  assert.deepEqual(groups.map(g => g.items.length), [2, 2, 0, 1])
  assert.deepEqual(groups[0].missing, ['Giới tính', 'Vòng eo'])
  assert.equal(filled, 5)
  assert.equal(total, 16)
  assert.equal(percent, 31)
  assert.equal(groupProfile([]).percent, 0)
})

test('classifies glycemic values against ADA cut-offs (glucose stored in mg/dL)', () => {
  assert.equal(glycemicBand(parseProfileLine(summary[2])), 'dm') // 129.7 mg/dL = 7.2 mmol/L
  assert.equal(glycemicBand(parseProfileLine(summary[3])), 'pre')
  assert.equal(glycemicBand(parseProfileLine('HbA1c: 5.2 % (bạn cung cấp)')), 'normal')
  assert.equal(glycemicBand(parseProfileLine('HbA1c: 6.5 % (bạn cung cấp)')), 'dm')
  assert.equal(glycemicBand(parseProfileLine(summary[0])), null)
})

const t = (msg_type, performative = 'REQUEST', receiver = 'X') => ({ msg_type, performative, receiver })

test('derives the 7-step plan for a full assessment and for a knowledge question', () => {
  const full = ['EXTRACT_FROM_TEXT', 'SCOPE_CHECK', 'PLAN_QUESTIONS', 'PREDICT_RISK', 'DELIBERATE', 'CHALLENGE_ML_VS_LABS', 'RECOMMEND_TREATMENT', 'COMPOSE_ASSESSMENT'].map(x => t(x))
  assert.deepEqual(planStatus(full), Array(7).fill('done'))
  const kb = ['EXTRACT_FROM_TEXT', 'SCOPE_CHECK', 'ANSWER_QUESTION', 'COMPOSE_KNOWLEDGE_ANSWER'].map(x => t(x))
  assert.deepEqual(planStatus(kb), ['done', 'done', 'skip', 'done', 'skip', 'skip', 'done'])
  assert.deepEqual(planStatus(full.slice(0, 4), true), ['done', 'done', 'done', 'now', 'todo', 'todo', 'todo'])
  assert.deepEqual(planStatus([]), Array(7).fill('todo'))
})

test('active agent is the receiver of the latest request or challenge', () => {
  assert.equal(activeAgent([t('PREDICT_RISK', 'REQUEST', 'PredictionAgent'), t('RISK_PREDICTION', 'INFORM', 'Orchestrator')]), 'PredictionAgent')
  assert.equal(activeAgent([t('X', 'CHALLENGE', 'KnowledgeAgent')]), 'KnowledgeAgent')
  assert.equal(activeAgent([]), null)
})
