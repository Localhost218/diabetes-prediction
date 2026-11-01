import { useState } from 'react'
import { Plus, Menu, Moon, Sun, HelpCircle, History, Trash2 } from 'lucide-react'

function Sidebar({ isDarkMode, setIsDarkMode, onOpenHelp, isOpen, onToggleSidebar, sessions, activeSessionId, onSelectSession, onNewConsultation, onDeleteSession }) {
  const [hoveredId, setHoveredId] = useState(null)
  return (
    <div className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <button className="icon-btn" onClick={onToggleSidebar} title="Đóng/mở sidebar">
          <Menu className="menu-icon" />
        </button>
      </div>

      {!isOpen && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '16px' }}>
          <button 
            className="icon-btn" 
            style={{ backgroundColor: 'var(--primary-color)', color: 'white' }} 
            onClick={onNewConsultation} 
            title="Cuộc trò chuyện mới"
          >
            <Plus size={20} />
          </button>
        </div>
      )}

      {isOpen && (
        <>
          <button className="new-chat-btn" onClick={onNewConsultation}>
            <Plus size={20} />
            New Consultation
          </button>

          <div className="history-section">
            <div className="history-title">HÔM NAY</div>
            
            {sessions && sessions.map(session => (
              <div 
                key={session.id} 
                className={`history-item ${activeSessionId === session.id ? 'active' : ''}`}
                onClick={() => onSelectSession(session.id)}
                onMouseEnter={() => setHoveredId(session.id)}
                onMouseLeave={() => setHoveredId(null)}
                style={{ 
                  cursor: 'pointer',
                  position: 'relative',
                  backgroundColor: activeSessionId === session.id ? 'var(--bg-hover)' : 'var(--bg-main)' 
                }}
              >
                <History size={16} className="shrink-0" />
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>
                  {session.title}
                </span>
                {hoveredId === session.id && (
                  <button
                    title="Xóa cuộc trò chuyện"
                    onClick={(e) => { e.stopPropagation(); onDeleteSession(session.id) }}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-secondary)', padding: '2px 4px',
                      display: 'flex', alignItems: 'center', borderRadius: 4,
                      flexShrink: 0,
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                    onMouseLeave={e => e.currentTarget.style.color = 'var(--text-secondary)'}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <div className="footer-item" onClick={() => setIsDarkMode(!isDarkMode)}>
              <div className="footer-item-left">
                {isDarkMode ? <Moon size={20} /> : <Sun size={20} />}
                <span>{isDarkMode ? 'Dark Mode' : 'Light Mode'}</span>
              </div>
              <label className="switch" onClick={(e) => e.stopPropagation()}>
                <input 
                  type="checkbox" 
                  checked={isDarkMode} 
                  onChange={() => setIsDarkMode(!isDarkMode)} 
                />
                <span className="slider"></span>
              </label>
            </div>
            
            <div className="footer-item" onClick={onOpenHelp}>
              <div className="footer-item-left">
                <HelpCircle size={20} />
                <span>Help Center</span>
              </div>
            </div>
          </div>
        </>
      )}

      {!isOpen && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '8px 0', gap: '4px' }}>
          {sessions && sessions.map(session => (
            <button
              key={session.id}
              className={`icon-btn ${activeSessionId === session.id ? 'icon-btn-active' : ''}`}
              onClick={() => onSelectSession(session.id)}
              title={session.title}
            >
              <History size={18} />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default Sidebar
