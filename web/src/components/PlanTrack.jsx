import { Check } from 'lucide-react'
import { PLAN_STAGES, planStatus } from '../services/workspace'

const STATUS_VI = { done: 'đã xong', now: 'đang chạy', skip: 'bỏ qua', todo: 'chưa tới' }

// Thanh "Kế hoạch lượt này": 7 bước của Orchestrator, suy ra từ thông điệp trên bus.
function PlanTrack({ steps = [], live = false }) {
  const status = planStatus(steps, live)
  const current = status.findIndex(s => s === 'now')
  const done = status.filter(s => s === 'done').length
  return (
    <section className="rail-card nb-plan" aria-label="Kế hoạch lượt này">
      <div className="nb-plan-head">
        <span className="nb-tag nb-ink">Kế hoạch lượt này</span>
        <span className="nb-tag">{!steps.length ? 'chưa chạy' : live ? `bước ${current + 1} / 7` : `xong ${done} / 7 bước`}</span>
      </div>
      <ol className="nb-track">
        {PLAN_STAGES.map((s, i) => (
          <li key={s.key} className={'is-' + status[i]} aria-current={status[i] === 'now' ? 'step' : undefined}
            title={`${s.label}: ${STATUS_VI[status[i]]}`}>
            <b>{status[i] === 'done' ? <Check size={12} aria-hidden="true" /> : status[i] === 'skip' ? '–' : i + 1}</b>
            <span>{s.label}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}

export default PlanTrack
