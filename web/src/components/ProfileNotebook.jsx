import { NotebookPen, User, FlaskConical, HeartPulse, RotateCcw, PencilLine, Gauge, Salad } from 'lucide-react'
import { groupProfile, glycemicBand } from '../services/workspace'

const ICONS = { about: User, labs: FlaskConical, diet: Salad, life: HeartPulse }
const BAND_VI = { normal: 'bình thường', pre: 'ngưỡng tiền ĐTĐ', dm: 'ngưỡng ĐTĐ' }

// Cột trái của "Hội chẩn": hồ sơ dạng sổ tay, mỗi chỉ số kèm nguồn; ô đường huyết tô theo ngưỡng ADA.
function ProfileNotebook({ profileSummary = [], onEdit, onShowResult, onRestart, disabled }) {
  const { groups, filled, total, percent } = groupProfile(profileSummary)
  return (
    <aside className="nb-notebook" aria-label="Hồ sơ bệnh nhân">
      <section className="rail-card nb-profile">
        <h2><NotebookPen size={17} aria-hidden="true" /> Hồ sơ bệnh nhân<span className="nb-count">{filled}/{total}</span></h2>
        {groups.map((g, gi) => {
          const Icon = ICONS[g.key]
          return (
            <div key={g.key} className="nb-group">
              <h3><Icon size={14} aria-hidden="true" />{String(gi + 1).padStart(2, '0')}. {g.title}
                <span className="nb-count">{g.filled}/{g.expected.length}</span></h3>
              <dl>
                {g.items.map(it => {
                  const band = glycemicBand(it)
                  return (
                    <div key={it.label} className={`nb-row${band ? ' band-' + band : ''}`} title={it.source + (band ? ' · ' + BAND_VI[band] : '')}>
                      <dt>{it.label}</dt>
                      <dd>{it.value}<span className="nb-src">{it.tag}</span></dd>
                    </div>
                  )
                })}
                {g.missing.map(label => (
                  <div key={label} className="nb-row is-missing"><dt>{label}</dt><dd>— chưa có</dd></div>
                ))}
              </dl>
            </div>
          )
        })}
      </section>
      <section className="rail-card nb-complete">
        <div className="nb-complete-head"><span className="nb-tag">Độ đầy đủ dữ liệu</span><strong>{percent} %</strong></div>
        <div className="nb-bar" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100} aria-label="Độ đầy đủ dữ liệu">
          <i style={{ width: percent + '%' }} />
        </div>
        <div className="nb-complete-actions">
          <button className="btn-secondary" onClick={onEdit} disabled={disabled}><PencilLine size={15} aria-hidden="true" /> Bổ sung</button>
          {onShowResult && <button className="btn-secondary nb-square" onClick={onShowResult} title="Xem kết quả đánh giá" aria-label="Xem kết quả đánh giá"><Gauge size={15} aria-hidden="true" /></button>}
          <button className="btn-secondary nb-square" onClick={onRestart} disabled={disabled} title="Bắt đầu lại" aria-label="Bắt đầu lại">
            <RotateCcw size={15} aria-hidden="true" />
          </button>
        </div>
      </section>
    </aside>
  )
}

export default ProfileNotebook
