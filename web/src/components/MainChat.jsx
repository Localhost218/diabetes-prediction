import { useState, useRef, useEffect } from 'react'
import { Plus, Mic, Send, Paperclip, Settings, ThumbsUp, ThumbsDown, Copy, Circle, Hourglass, Lightbulb, Camera, Menu, FileText, X, RotateCcw, Cpu } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { transcribeAudio } from '../services/llmService'
import CameraModal from './CameraModal'
import ConsultationDetails from './ConsultationDetails'
import ThinkingPanel from './ThinkingPanel'

const MAX_FILES = 5
const ACCEPT = '.pdf,.docx,.csv,.txt,image/jpeg,image/png,image/webp'

const SUGGESTIONS = [
  'Tôi 52 tuổi, nam, cao 1m65 nặng 72kg, đường huyết lúc đói 7.2 mmol/L, bố bị tiểu đường. Nguy cơ của tôi thế nào?',
  'HbA1c bao nhiêu thì được coi là tiền đái tháo đường?',
  'Mẹ tôi 61 tuổi, BMI 29, có uống thuốc huyết áp, HbA1c 6.1%. Bà có cần đi khám không?',
  'Người tiền đái tháo đường nên vận động và ăn uống như thế nào?',
]

function Attachment({ a }) {
  const url = a.localUrl || (a.document_id ? `/api/documents/${a.document_id}/content` : null)
  if (a.mime_type?.startsWith('image/') && url) {
    return <a href={url} target="_blank" rel="noreferrer"><img src={url} alt={a.filename} className="attachment-image" /></a>
  }
  return (
    <a className="attachment-chip" href={url || undefined} target="_blank" rel="noreferrer">
      <FileText size={14} /> <span>{a.filename}</span>
    </a>
  )
}

function MainChat({ onOpenSettings, activeSession, onSendMessage, onResetProfile, onToggleSidebar, systemInfo }) {
  const [showPlusMenu, setShowPlusMenu] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [toastMsg, setToastMsg] = useState('')
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const toastTimerRef = useRef(null)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [feedbackState, setFeedbackState] = useState({}) // { [msgId]: 'liked' | 'disliked' | null } – optimistic UI
  const [feedbackLoading, setFeedbackLoading] = useState({}) // { [msgId]: true } – prevent double-click

  const menuRef = useRef(null)
  const fileInputRef = useRef(null)
  const chatEndRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeSession?.messages])

  // Persist feedback to backend
  const handleFeedback = async (msg, type) => {
    if (!msg.id || feedbackLoading[msg.id]) return

    // Determine new state (toggle off if same, switch otherwise)
    const current = feedbackState[msg.id] !== undefined
      ? feedbackState[msg.id]
      : msg.isHelpful ? 'liked' : msg.isUseless ? 'disliked' : null
    const next = current === type ? null : type

    // Optimistic update
    setFeedbackState(prev => ({ ...prev, [msg.id]: next }))
    setFeedbackLoading(prev => ({ ...prev, [msg.id]: true }))

    try {
      await fetch(`/api/sessions/${activeSession.id}/messages/${msg.id}/feedback`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ isHelpful: next === 'liked', isUseless: next === 'disliked' })
      })
    } catch (err) {
      console.error('Feedback error:', err)
      // Rollback optimistic update on error
      setFeedbackState(prev => ({ ...prev, [msg.id]: current }))
    } finally {
      setFeedbackLoading(prev => ({ ...prev, [msg.id]: false }))
    }
  }

  const showToast = (msg) => {
    setToastMsg(msg)
    clearTimeout(toastTimerRef.current)
    toastTimerRef.current = setTimeout(() => setToastMsg(''), 3000)
  }

  const addFiles = (files) => {
    if (selectedFiles.length + files.length > MAX_FILES) {
      showToast(`Chỉ được đính kèm tối đa ${MAX_FILES} file mỗi tin nhắn.`)
      return
    }
    setSelectedFiles(prev => [...prev, ...files])
  }

  // MediaRecorder-based mic (works on Brave — does NOT use Web Speech API)
  const toggleListening = async () => {
    if (isTranscribing) return

    if (isListening) {
      mediaRecorderRef.current?.stop()
      setIsListening(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []

      // Pick a format; prefer ogg (natively supported by Gemini inline_data)
      const mimeType = ['audio/ogg;codecs=opus', 'audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
        .find(t => MediaRecorder.isTypeSupported(t)) || ''

      // Strip codec suffix: Gemini requires bare mime type (e.g. 'audio/webm', NOT 'audio/webm;codecs=opus')
      const geminiMimeType = mimeType.split(';')[0] || 'audio/webm'

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {})
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: geminiMimeType })
        setIsTranscribing(true)
        try {
          const text = await transcribeAudio(blob)
          if (text) {
            setInputValue(prev => (prev ? prev + ' ' : '') + text)
            showToast('Đã điền transcript vào ô chat. Bạn có thể sửa lại rồi bấm Gửi.')
          }
        } catch (err) {
          console.error('Transcription failed:', err)
          showToast(`❌ ${err.message}`)
        } finally {
          setIsTranscribing(false)
        }
      }

      recorder.start()
      setIsListening(true)
    } catch (err) {
      console.error('Mic access error:', err)
      alert('Không thể truy cập microphone. Vui lòng kiểm tra quyền trình duyệt.')
    }
  }

  // Close menus on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowPlusMenu(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const sendText = async (text, files = []) => {
    if (isSending) return
    setIsSending(true)
    try {
      await onSendMessage(text, files)
    } finally {
      setIsSending(false)
    }
  }

  const handleSend = async () => {
    if (!inputValue.trim() && selectedFiles.length === 0) return;
    const text = inputValue;
    const files = selectedFiles;
    setInputValue('');
    setSelectedFiles([]);
    await sendText(text, files)
  };

  const llmLabel = systemInfo
    ? (systemInfo.llm.enabled ? `${systemInfo.llm.provider} · ${systemInfo.llm.model}` : 'Chế độ luật (không LLM)')
    : null

  return (
    <div className="main-content">
      <div className="header" style={{ position: 'relative' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button className="mobile-menu-btn icon-btn" onClick={onToggleSidebar} title="Mở menu" aria-label="Mở sidebar">
            <Menu size={20} />
          </button>
          <div className="brand">
            <img src="/logo.png" alt="" />
            <span>Diabetes Prediction</span>
          </div>
        </div>
        {activeSession?.title && (
          <div className="session-title-header" style={{
            position: 'absolute', left: '50%', transform: 'translateX(-50%)',
            fontWeight: 600, fontSize: '1rem',
            color: 'var(--text-primary)',
            maxWidth: '40%', overflow: 'hidden',
            whiteSpace: 'nowrap', textOverflow: 'ellipsis',
            pointerEvents: 'none'
          }}>
            {activeSession.title}
          </div>
        )}
        <div className="header-actions">
          {llmLabel && (
            <span className="system-badge" title={`${systemInfo.agents.length} tác tử · KG: ${systemInfo.knowledge_graph} · Embedding: ${systemInfo.embedding}`}>
              <Cpu size={14} /> {llmLabel}
            </span>
          )}
          <button className="icon-btn" onClick={onResetProfile} title="Xóa dữ liệu sức khỏe đã ghi nhận trong cuộc trò chuyện">
            <RotateCcw size={18} />
          </button>
        </div>
      </div>

      {/* Toast notification */}
      {toastMsg && (
        <div style={{
          position: 'fixed', bottom: 100, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(30,30,30,0.92)', color: '#fff',
          padding: '10px 20px', borderRadius: 24,
          fontSize: '0.85rem', zIndex: 9999,
          boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
          animation: 'fadeIn 0.2s ease-out'
        }}>
          {toastMsg}
        </div>
      )}

      <div className="chat-area">
        <div className="chat-container">
          {(!activeSession || !activeSession.messages || activeSession.messages.length === 0) ? (
            <div className="welcome-screen">
              <img src="/logo.png" alt="Diabetes Prediction" className="welcome-logo" />
              <h1 className="welcome-heading">Sàng lọc nguy cơ đái tháo đường cùng hội đồng AI.</h1>
              <p className="welcome-sub">Nhập chỉ số sức khỏe, hỏi kiến thức, hoặc đính kèm phiếu xét nghiệm / bệnh án<br/>(PDF, ảnh, DOCX, CSV) để các tác tử phân tích.</p>
              <div className="suggestions-grid">
                {SUGGESTIONS.map((prompt, i) => (
                  <button key={i} className="suggestion-card" onClick={() => sendText(prompt)}>
                    <Lightbulb size={18} className="suggestion-icon" />
                    <span>{prompt}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            activeSession.messages.map((msg, idx) => (
              <div key={msg.id || idx} className={`message ${msg.sender}`}>
                {msg.sender === 'bot' && (
                  <div className="bot-avatar">
                    <img src="/logo.png" alt="Diabetes Prediction" style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '50%' }} />
                  </div>
                )}
                <div className={msg.sender === 'bot' ? 'bot-content' : undefined}>
                  {msg.sender === 'bot' && msg._loading && (
                    <ThinkingPanel live steps={msg._steps} graph={msg._graph} />
                  )}
                  {msg.sender === 'bot' && !msg._loading && msg.data?.transcript && (
                    <ThinkingPanel steps={msg.data.transcript} graph={msg.data.graph} seconds={msg._thinkSeconds} />
                  )}
                  {!(msg.sender === 'bot' && msg._loading) && (
                  <div className="message-bubble" style={{ color: msg.is_error ? 'red' : 'inherit' }}>
                    {msg.attachments?.length > 0 && (
                      <div className="attachments">
                        {msg.attachments.map((a, i) => <Attachment key={i} a={a} />)}
                      </div>
                    )}
                    {msg._loading ? (
                      <span className="thinking">
                        <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
                        {msg._hasFiles ? 'Đang đọc bệnh án và hội chẩn…' : 'Hội đồng các tác tử đang phân tích…'}
                      </span>
                    ) : msg.sender === 'bot' ? (
                      <div className="markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                      </div>
                    ) : msg.text}
                  </div>
                  )}
                  {msg.sender === 'bot' && !msg._loading && <ConsultationDetails data={msg.data} />}
                  {msg.sender === 'bot' && !msg._loading && msg.id && (
                    <div className="message-actions">
                      {(() => {
                        const isLiked = feedbackState[msg.id] !== undefined ? feedbackState[msg.id] === 'liked' : msg.isHelpful
                        const isDisliked = feedbackState[msg.id] !== undefined ? feedbackState[msg.id] === 'disliked' : msg.isUseless
                        return (
                          <>
                            <button
                              className={`action-btn${isLiked ? ' liked' : ''}`}
                              onClick={() => handleFeedback(msg, 'liked')}
                              disabled={!!feedbackLoading[msg.id]}
                              title="Hữu ích"
                            >
                              <ThumbsUp size={16} fill={isLiked ? 'currentColor' : 'none'} />
                            </button>
                            <button
                              className={`action-btn${isDisliked ? ' disliked' : ''}`}
                              onClick={() => handleFeedback(msg, 'disliked')}
                              disabled={!!feedbackLoading[msg.id]}
                              title="Không hữu ích"
                            >
                              <ThumbsDown size={16} fill={isDisliked ? 'currentColor' : 'none'} />
                            </button>
                          </>
                        )
                      })()}
                      <button
                        className="action-btn"
                        onClick={() => {
                          navigator.clipboard.writeText(msg.text)
                          showToast('Đã copy tin nhắn!')
                        }}
                        title="Copy"
                      >
                        <Copy size={16} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      <div className="input-area">
        <div className="input-container">
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            multiple
            accept={ACCEPT}
            onChange={(e) => {
              if (e.target.files.length > 0) addFiles(Array.from(e.target.files))
              e.target.value = ''
              setShowPlusMenu(false);
            }}
          />
          {showCamera && (
            <CameraModal
              onClose={() => setShowCamera(false)}
              onCapture={(file) => {
                addFiles([file])
                setShowCamera(false);
              }}
            />
          )}
          {selectedFiles.length > 0 && (
            <div className="selected-files">
              {selectedFiles.map((file, idx) => (
                <div key={idx} className="selected-file">
                  {file.type.startsWith('image/')
                    ? <img src={URL.createObjectURL(file)} alt="preview" />
                    : <span className="selected-file-name"><FileText size={14} /> {file.name}</span>}
                  <button onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== idx))} title="Bỏ file">
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
          {showPlusMenu && (
            <div className="plus-menu" ref={menuRef}>
              <div className="menu-item" onClick={() => fileInputRef.current.click()}>
                <Paperclip size={18} />
                <span>Tải bệnh án / phiếu xét nghiệm (PDF, ảnh, DOCX, CSV)</span>
              </div>
              <div className="menu-item" onClick={() => {
                setShowCamera(true);
                setShowPlusMenu(false);
              }}>
                <Camera size={18} />
                <span>Chụp ảnh phiếu xét nghiệm</span>
              </div>
              <div className="menu-item" onClick={() => {
                setShowPlusMenu(false)
                onOpenSettings()
              }}>
                <Settings size={18} />
                <span>Cài đặt</span>
              </div>
            </div>
          )}

          <div className="input-wrapper">
            <button
              className="icon-btn"
              onClick={() => setShowPlusMenu(!showPlusMenu)}
              title="Đính kèm bệnh án"
            >
              <Plus size={20} />
            </button>
            <input
              type="text"
              placeholder="Nhập chỉ số sức khỏe hoặc câu hỏi cho Diabetes Prediction..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isSending && (inputValue.trim() || selectedFiles.length > 0)) {
                  handleSend();
                }
              }}
            />
            <button className={`icon-btn ${isListening ? 'recording' : ''}`} onClick={toggleListening}
              disabled={isTranscribing}
              title={isListening ? 'Dừng ghi âm' : isTranscribing ? 'Đang xử lý...' : 'Ghi âm giọng nói'}
            >
              {isTranscribing
                ? <Hourglass size={20} className="spin-hourglass" />
                : isListening
                ? <Circle size={20} fill="currentColor" className="record-icon" />
                : <Mic size={20} />}
            </button>
            <button
              className="icon-btn"
              disabled={isSending || (!inputValue.trim() && selectedFiles.length === 0)}
              onClick={handleSend}
            >
              <Send size={20} />
            </button>
          </div>

          <div className="disclaimer">
            Diabetes Prediction chỉ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ.
          </div>
        </div>
      </div>
    </div>
  )
}

export default MainChat
