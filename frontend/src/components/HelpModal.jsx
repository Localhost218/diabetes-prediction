import { X } from 'lucide-react'

function HelpModal({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          <X size={20} />
        </button>
        
        <div className="help-content">
          <div className="help-logo">
            Help Center
          </div>
          
          <div style={{ margin: '24px 0', display: 'flex', justifyContent: 'center' }}>
             <div className="brand brand-large">
               <img src="/logo.png" alt="" />
               <span>Diabetes Prediction</span>
             </div>
          </div>

          <div className="help-title">Trợ lý chatbot của Diabetes Prediction</div>
          <div className="help-author">
            Develop by <strong>Nguyễn Tuấn Anh</strong> — A* Squad Member
          </div>
        </div>
      </div>
    </div>
  )
}

export default HelpModal
