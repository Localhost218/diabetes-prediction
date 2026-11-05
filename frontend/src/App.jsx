import { useState, useEffect } from 'react'

// Nhịp hiện từng bước suy luận: luồng không LLM xong trong ~1 giây, nên giãn ra cho người xem kịp đọc
const STEP_MS = 140
const sleep = (ms) => new Promise(r => setTimeout(r, ms))
import Sidebar from './components/Sidebar'
import MainChat from './components/MainChat'
import HelpModal from './components/HelpModal'
import VoiceSettingsModal from './components/VoiceSettingsModal'
import AgentRoom from './components/AgentRoom'
import { sendChatStream, fetchSystemInfo, resetProfile } from './services/llmService'

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [apiError, setApiError] = useState(null)
  const [isConnecting, setIsConnecting] = useState(true)
  const [systemInfo, setSystemInfo] = useState(null)
  const [roomOpen, setRoomOpen] = useState(() => window.innerWidth > 1180)
  const [roomMessageId, setRoomMessageId] = useState(null) // null → tin nhắn bot mới nhất

  // State management for sessions
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [activeSession, setActiveSession] = useState(null)

  const loadSessions = () => {
    setIsConnecting(true)
    setApiError(null)
    fetch('/api/sessions')
      .then(res => {
        if (!res.ok) throw new Error(`Server lỗi ${res.status} - vui lòng thử lại sau`)
        return res.json()
      })
      .then(data => {
        setIsConnecting(false)
        setSessions(data)
        if (data.length > 0) setActiveSessionId(data[0].id)
        else handleNewConsultation()
      })
      .catch(err => {
        setIsConnecting(false)
        setApiError(err.message || 'Không thể kết nối đến máy chủ')
      })
  }

  useEffect(() => {
    loadSessions()
    fetchSystemInfo().then(setSystemInfo).catch(() => setSystemInfo(null))
  }, [])

  useEffect(() => {
    if (activeSessionId) {
      fetch(`/api/sessions/${activeSessionId}`)
        .then(res => res.json())
        .then(data => setActiveSession(data))
        .catch(err => console.error("Error fetching session:", err))
    }
  }, [activeSessionId])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light')
  }, [isDarkMode])

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const handleNewConsultation = async () => {
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Cuộc trò chuyện mới' })
      })
      const newSession = await res.json()
      setSessions(prev => [newSession, ...prev])
      setActiveSessionId(newSession.id)
      setActiveSession(newSession)
    } catch (err) {
      console.error("Error creating session:", err)
    }
  }

  const handleDeleteSession = async (sessionId) => {
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' })
      const remaining = sessions.filter(s => s.id !== sessionId)
      setSessions(remaining)
      if (activeSessionId === sessionId) {
        if (remaining.length > 0) setActiveSessionId(remaining[0].id)
        else handleNewConsultation()
      }
    } catch (err) {
      console.error("Error deleting session:", err)
    }
  }

  const handleResetProfile = async () => {
    if (!activeSessionId) return
    if (!window.confirm('Xóa toàn bộ chỉ số sức khỏe đã ghi nhận trong cuộc trò chuyện này?')) return
    await resetProfile(activeSessionId)
    setActiveSession(prev => ({ ...prev, profile_summary: [] }))
  }

  const handleSendMessage = async (text, files = []) => {
    if (!activeSessionId) return;

    // 1. Hiện ngay tin nhắn người dùng + trạng thái đang xử lý
    const tempUserMsg = {
      sender: 'user', text, is_error: false,
      attachments: files.map(f => ({ filename: f.name, mime_type: f.type, localUrl: URL.createObjectURL(f) }))
    }
    const tempId = `pending-${Date.now()}`
    const tempBotMsg = { sender: 'bot', text: '', _loading: true, _hasFiles: files.length > 0, _tempId: tempId,
                         _steps: [], _graph: null }
    setRoomMessageId(null)
    setActiveSession(prev => ({
      ...prev,
      messages: [...(prev?.messages || []), tempUserMsg, tempBotMsg]
    }))

    // Bước suy luận tới dồn vào hàng đợi, nhả ra đều tay để khối "đang suy luận" chạy mượt
    const queue = []
    let shown = 0
    const patchTemp = (fn) => setActiveSession(prev => ({
      ...prev, messages: (prev?.messages || []).map(m => (m._tempId === tempId ? fn(m) : m))
    }))
    const ticker = setInterval(() => {
      if (shown >= queue.length) return
      const ev = queue[shown++]
      patchTemp(m => (ev.type === 'graph' ? { ...m, _graph: ev } : { ...m, _steps: [...m._steps, ev] }))
    }, STEP_MS)
    const started = performance.now()

    try {
      // 2. Backend chạy một lượt hội chẩn đa tác tử, vừa chạy vừa gửi từng thông điệp giữa các tác tử
      const result = await sendChatStream(activeSessionId, text, files, ev => queue.push(ev))
      while (shown < queue.length) await sleep(STEP_MS)   // cho khối suy luận hiện hết rồi mới hiện câu trả lời
      clearInterval(ticker)
      const seconds = Math.round((performance.now() - started) / 100) / 10
      setActiveSession(prev => ({
        ...prev,
        title: result.session.title,
        messages: [...(prev?.messages || []).filter(m => m !== tempUserMsg && m._tempId !== tempId),
                   result.user_message, { ...result.bot_message, _thinkSeconds: seconds }]
      }))
      setSessions(prev => [
        { ...result.session },
        ...prev.filter(s => s.id !== result.session.id)
      ])
    } catch (err) {
      clearInterval(ticker)
      console.error("Error sending message:", err)
      setActiveSession(prev => ({
        ...prev,
        messages: (prev?.messages || []).filter(m => m._tempId !== tempId)
          .concat([{ sender: 'bot', text: `Gửi tin nhắn thất bại: ${err.message}`, is_error: true }])
      }))
    }
  }

  // Tin nhắn hiển thị trong Phòng hội chẩn: tin được chọn, hoặc tin bot mới nhất có transcript
  const messages = activeSession?.messages || []
  const roomLoading = messages.length > 0 && messages[messages.length - 1]._loading
  const roomMessage = messages.find(m => roomMessageId && m.id === roomMessageId)
    || [...messages].reverse().find(m => m.sender === 'bot' && m.data?.transcript)

  const showInRoom = (msg) => {
    setRoomMessageId(msg.id || null)
    setRoomOpen(true)
  }

  return (
    <div className="app-container">
      {/* Connection status banners */}
      {isConnecting && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999,
          background: 'linear-gradient(90deg, #1e3a5f, #2563eb)',
          color: '#fff', padding: '10px 20px',
          display: 'flex', alignItems: 'center', gap: '10px',
          fontSize: '0.88rem', fontWeight: 500, boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
        }}>
          <span style={{ display: 'inline-block', width: 14, height: 14, border: '2px solid #fff',
            borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          Đang kết nối đến máy chủ…
        </div>
      )}
      {apiError && !isConnecting && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999,
          background: '#dc2626', color: '#fff', padding: '10px 20px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          fontSize: '0.88rem', fontWeight: 500, boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
        }}>
          <span>⚠️ {apiError}</span>
          <button
            onClick={loadSessions}
            style={{
              background: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.5)',
              color: '#fff', padding: '4px 14px', borderRadius: 6, cursor: 'pointer',
              fontSize: '0.82rem', fontWeight: 600
            }}
          >Thử lại</button>
        </div>
      )}

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div className="mobile-overlay" onClick={() => setSidebarOpen(false)}></div>
      )}

      <Sidebar
        isDarkMode={isDarkMode}
        setIsDarkMode={setIsDarkMode}
        onOpenHelp={() => setShowHelp(true)}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onToggleSidebar={toggleSidebar}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewConsultation={handleNewConsultation}
        onDeleteSession={handleDeleteSession}
      />
      <MainChat
        key={activeSessionId}
        onOpenSettings={() => setShowSettings(true)}
        onToggleSidebar={toggleSidebar}
        activeSession={activeSession}
        onSendMessage={handleSendMessage}
        onResetProfile={handleResetProfile}
        systemInfo={systemInfo}
        roomOpen={roomOpen}
        onToggleRoom={() => setRoomOpen(o => !o)}
        onShowInRoom={showInRoom}
      />
      {roomOpen && (
        <AgentRoom key={roomMessage?.id || (roomLoading ? 'loading' : 'none')} message={roomMessage} loading={roomLoading} onClose={() => setRoomOpen(false)} />
      )}

      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
      {showSettings && <VoiceSettingsModal onClose={() => setShowSettings(false)} />}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

export default App
