import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

// Hộp thoại theo theme sổ tay: Esc hoặc bấm nền để đóng, focus vào hộp khi mở.
function NbModal({ title, subtitle = '', icon: Icon = null, onClose, wide = false, actions = null, children }) {
  const ref = useRef(null)
  useEffect(() => {
    ref.current?.focus()
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="nb-modal-overlay" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className={`nb-modal${wide ? ' is-wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={ref}>
        <div className="nb-modal-head">
          {Icon && <span className="nb-modal-icon"><Icon size={18} aria-hidden="true" /></span>}
          <div className="nb-modal-title"><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>
          <div className="nb-modal-actions">
            {actions}
            <button className="nb-modal-close" onClick={onClose} aria-label="Đóng" title="Đóng (Esc)"><X size={16} /></button>
          </div>
        </div>
        <div className="nb-modal-body">{children}</div>
      </div>
    </div>
  )
}

export default NbModal
