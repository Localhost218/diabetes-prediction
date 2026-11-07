import { useState } from 'react'
import { X, ChevronDown, Volume2, Save } from 'lucide-react'

function VoiceSettingsModal({ onClose }) {
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [gender, setGender] = useState('Nữ')
  const [age, setAge] = useState('Trẻ')
  const [pitch, setPitch] = useState('Vừa')
  const [style, setStyle] = useState('Bình thường')

  const profileLabel = `${gender === 'Nữ' ? 'Nữ' : 'Nam'} · ${age} · Tông ${pitch.toLowerCase()} · ${style}`

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '450px' }}>
        <button className="modal-close" onClick={onClose}>
          <X size={20} />
        </button>
        
        <div className="voice-settings-header">
          <h2 className="voice-settings-title">Cài đặt giọng đọc</h2>
          <p className="voice-settings-subtitle">Điều chỉnh cấu hình Text-to-Speech</p>
        </div>

        <div className="setting-row">
          <div className="setting-info">
            <h4>Text-to-Speech</h4>
            <p>Tinh chỉnh giọng đọc cho phản hồi của chatbot. Cấu hình này chỉ áp dụng cho tiếng Việt.</p>
          </div>
          
          <label className="switch">
            <input 
              type="checkbox" 
              checked={ttsEnabled} 
              onChange={() => setTtsEnabled(!ttsEnabled)} 
            />
            <span className="slider"></span>
          </label>
        </div>

        <div className="setting-grid" style={{ opacity: ttsEnabled ? 1 : 0.5, pointerEvents: ttsEnabled ? 'auto' : 'none' }}>
          <div className="setting-field">
            <label>Giới tính</label>
            <div className="select-wrapper">
              <select value={gender} onChange={e => setGender(e.target.value)}>
                <option value="Nữ">Nữ</option>
                <option value="Nam">Nam</option>
              </select>
              <ChevronDown size={16} className="select-icon" />
            </div>
          </div>

          <div className="setting-field">
            <label>Độ tuổi</label>
            <div className="select-wrapper">
              <select value={age} onChange={e => setAge(e.target.value)}>
                <option value="Trẻ">Trẻ</option>
                <option value="Trưởng thành">Trưởng thành</option>
                <option value="Già">Già</option>
              </select>
              <ChevronDown size={16} className="select-icon" />
            </div>
          </div>

          <div className="setting-field">
            <label>Tông giọng</label>
            <div className="select-wrapper">
              <select value={pitch} onChange={e => setPitch(e.target.value)}>
                <option value="Cao">Cao</option>
                <option value="Vừa">Vừa</option>
                <option value="Thấp">Thấp</option>
              </select>
              <ChevronDown size={16} className="select-icon" />
            </div>
          </div>

          <div className="setting-field">
            <label>Kiểu nói</label>
            <div className="select-wrapper">
              <select value={style} onChange={e => setStyle(e.target.value)}>
                <option value="Bình thường">Bình thường</option>
                <option value="Nhanh">Nhanh</option>
                <option value="Chậm">Chậm</option>
              </select>
              <ChevronDown size={16} className="select-icon" />
            </div>
          </div>
        </div>

        {/* Voice profile preview card */}
        <div className="voice-profile-card">
          <div className="voice-profile-icon">
            <Volume2 size={18} />
          </div>
          <div className="voice-profile-info">
            <span className="voice-profile-label">Voice Profile</span>
            <span className="voice-profile-value">{profileLabel}</span>
          </div>
        </div>

        <button className="save-settings-btn" onClick={onClose}>
          <Save size={16} />
          Lưu cài đặt
        </button>
      </div>
    </div>
  )
}

export default VoiceSettingsModal
