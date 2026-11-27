import { useState, useEffect, useCallback, useRef } from 'react'
import MainChat from './components/MainChat'
import HelpModal from './components/HelpModal'
import VoiceSettingsModal from './components/VoiceSettingsModal'
import { sendChatStream, fetchSystemInfo } from './services/llmService'

// Nhịp hiện từng bước suy luận: luồng không LLM xong trong ~1 giây, nên giãn ra cho người xem kịp đọc
const STEP_MS = 140
const sleep = (ms) => new Promise(r => setTimeout(r, ms))

// Phiên chỉ sống trong một lần mở trang: không có lịch sử, đóng/tải lại trang hoặc "Bắt đầu lại" là xóa phiên cũ.
const createSession = async () => {
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'Tư vấn mới' })
  })
  if (!res.ok) throw new Error(`Không thể kết nối máy chủ (${res.status})`)
  return res.json()
}
const discardSession = (id) => {
  if (id) fetch(`/api/sessions/${id}`, { method: 'DELETE', keepalive: true }).catch(() => {})
}

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false)  // mặc định giao diện sáng; người dùng tự bật tối nếu muốn
  const [showHelp, setShowHelp] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [apiError, setApiError] = useState(null)
  const [isConnecting, setIsConnecting] = useState(true)
  const [systemInfo, setSystemInfo] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [isSending, setIsSending] = useState(false)
  const sendingRef = useRef(false)
  const sessionIdRef = useRef(null)
  const startedRef = useRef(false)

  const startSession = useCallback(async () => {
    setIsConnecting(true)
    setApiError(null)
    discardSession(sessionIdRef.current)
    sessionIdRef.current = null
    setActiveSession(null)
    try {
      const session = await createSession()
      sessionIdRef.current = session.id
      setActiveSession({ ...session, messages: [], profile_summary: [] })
    } catch (err) {
      setApiError(err.message || 'Không thể kết nối đến máy chủ')
    } finally {
      setIsConnecting(false)
    }
  }, [])

  useEffect(() => {
    if (startedRef.current) return   // StrictMode gọi effect 2 lần ở dev: chỉ tạo một phiên
    startedRef.current = true
    startSession()
    fetchSystemInfo().then(setSystemInfo).catch(() => setSystemInfo(null))
    const onLeave = () => discardSession(sessionIdRef.current)
    window.addEventListener('pagehide', onLeave)
    return () => window.removeEventListener('pagehide', onLeave)
  }, [startSession])

  // Esc đóng hộp thoại đang mở
  useEffect(() => {
    if (!showHelp && !showSettings) return
    const onKey = (e) => { if (e.key === 'Escape') { setShowHelp(false); setShowSettings(false) } }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [showHelp, showSettings])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light')
  }, [isDarkMode])

  const handleRestart = () => {
    if (sendingRef.current) return
    if (activeSession?.messages?.length && !window.confirm('Bắt đầu lại sẽ xóa toàn bộ chỉ số và cuộc trò chuyện hiện tại. Tiếp tục?')) return
    startSession()
  }

  const handleSendMessage = async (text, files = []) => {
    const sessionId = activeSession?.id
    if (!sessionId || sendingRef.current) return null
    sendingRef.current = true
    setIsSending(true)

    // 1. Hiện ngay tin nhắn người dùng + trạng thái đang xử lý
    const tempUserMsg = {
      sender: 'user', text, is_error: false,
      attachments: files.map(f => ({ filename: f.name, mime_type: f.type, localUrl: URL.createObjectURL(f) }))
    }
    const tempId = `pending-${Date.now()}`
    const tempBotMsg = { sender: 'bot', text: '', _loading: true, _hasFiles: files.length > 0, _tempId: tempId,
                         _steps: [], _graph: null }
    setActiveSession(prev => ({ ...prev, messages: [...(prev?.messages || []), tempUserMsg, tempBotMsg] }))

    // Bước suy luận tới dồn vào hàng đợi, nhả ra đều tay để khối "đang suy luận" chạy mượt
    const queue = []
    let shown = 0
    const patchTemp = (fn) => setActiveSession(prev => prev?.id !== sessionId ? prev : ({
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
      const result = await sendChatStream(sessionId, text, files, ev => {
        // Hồ sơ cập nhật ngay khi tiếp nhận xong, không chờ hội đồng
        if (ev.type === 'profile') setActiveSession(prev => prev?.id !== sessionId ? prev : ({ ...prev, profile_summary: ev.summary }))
        else queue.push(ev)
      })
      while (shown < queue.length) await sleep(STEP_MS)   // cho khối suy luận hiện hết rồi mới hiện câu trả lời
      clearInterval(ticker)
      const seconds = Math.round((performance.now() - started) / 100) / 10
      setActiveSession(prev => prev?.id !== sessionId ? prev : ({
        ...prev,
        title: result.session.title,
        profile_summary: result.bot_message.data?.profile_summary || prev.profile_summary,
        messages: [...(prev?.messages || []).filter(m => m !== tempUserMsg && m._tempId !== tempId),
                   result.user_message, { ...result.bot_message, _thinkSeconds: seconds }]
      }))
      return result
    } catch (err) {
      clearInterval(ticker)
      console.error('Error sending message:', err)
      setActiveSession(prev => prev?.id !== sessionId ? prev : ({
        ...prev,
        messages: (prev?.messages || []).filter(m => m._tempId !== tempId)
          .concat([{ sender: 'bot', text: `Gửi tin nhắn thất bại: ${err.message}`, is_error: true }])
      }))
      return null
    } finally {
      sendingRef.current = false
      setIsSending(false)
    }
  }

  return (
    <div className="app-container">
      {isConnecting && (
        <div className="connection-banner" role="status">
          <span className="connection-spinner" aria-hidden="true" />
          Đang chuẩn bị phiên tư vấn…
        </div>
      )}
      {apiError && !isConnecting && (
        <div className="connection-banner is-error" role="alert">
          <span>⚠️ {apiError}</span>
          <button onClick={startSession}>Thử lại</button>
        </div>
      )}

      <MainChat
        key={activeSession?.id || 'loading'}
        activeSession={activeSession}
        isSending={isSending}
        onSendMessage={handleSendMessage}
        onProfileSaved={summary => setActiveSession(prev => ({ ...prev, profile_summary: summary }))}
        onRestart={handleRestart}
        onOpenSettings={() => setShowSettings(true)}
        onOpenHelp={() => setShowHelp(true)}
        isDarkMode={isDarkMode}
        onToggleTheme={() => setIsDarkMode(d => !d)}
        systemInfo={systemInfo}
      />

      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
      {showSettings && <VoiceSettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  )
}

export default App
