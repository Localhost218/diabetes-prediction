import { useState, useRef, useEffect, useCallback } from 'react'
import { Plus, Mic, Send, Paperclip, Settings, ThumbsUp, ThumbsDown, Copy, Circle, Hourglass, Camera, FileText, X, RotateCcw, Cpu, Moon, Sun, HelpCircle, ClipboardList, Gauge, Database, Bot } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { transcribeAudio, saveProfile } from '../services/llmService'
import CameraModal from './CameraModal'
import ConsultationDetails from './ConsultationDetails'
import ThinkingPanel from './ThinkingPanel'
import PredictionPanel from './PredictionPanel'
import AgentFlow from './AgentFlow'
import NbModal from './NbModal'
import ProfileNotebook from './ProfileNotebook'
import PlanTrack from './PlanTrack'
import KnowledgeView from './KnowledgeView'
import AgentsView from './AgentsView'
import './ConsultationWorkspace.css'
import '../theme-notebook.css'

const MAX_FILES = 5
const ACCEPT = '.pdf,.docx,.csv,.txt,image/jpeg,image/png,image/webp'

const SUGGESTIONS = [
  { label: 'HbA1c bao nhiêu là tiền đái tháo đường?', text: 'HbA1c bao nhiêu thì được coi là tiền đái tháo đường?' },
  { label: 'Dấu hiệu sớm của bệnh tiểu đường', text: 'Những dấu hiệu sớm của bệnh tiểu đường là gì?' },
  { label: 'Tiền đái tháo đường nên ăn gì?', text: 'Người tiền đái tháo đường nên vận động và ăn uống như thế nào?' },
  { label: 'Ví dụ: tự mô tả chỉ số', text: 'Tôi 52 tuổi, nam, cao 1m65 nặng 72kg, đường huyết lúc đói 7.2 mmol/L, bố bị tiểu đường. Nguy cơ của tôi thế nào?' },
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

function MainChat({ onProfileSaved, onOpenSettings, onOpenHelp, activeSession, onSendMessage, onRestart, isDarkMode, onToggleTheme, systemInfo, isSending }) {
  // Mới vào: mở sẵn form nhập hồ sơ (có thể bỏ qua). 'form' | 'result' | 'kb' | 'agents' | null
  const [modal, setModal] = useState('form')
  const closeModal = useCallback(() => setModal(null), [])
  const [showPlusMenu, setShowPlusMenu] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [toastMsg, setToastMsg] = useState('')
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const toastTimerRef = useRef(null)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [feedbackState, setFeedbackState] = useState({}) // { [msgId]: 'liked' | 'disliked' | null } – optimistic UI
  const [feedbackLoading, setFeedbackLoading] = useState({}) // { [msgId]: true } – prevent double-click

  const menuRef = useRef(null)
  const fileInputRef = useRef(null)
  const chatAreaRef = useRef(null)

  // Tự cuộn xuống cuối chỉ khi người dùng đang ở cuối: khối suy luận cập nhật mỗi 140 ms,
  // cuộn cưỡng bức sẽ giật ngược mỗi lần người dùng lăn chuột lên đọc
  const stickRef = useRef(true)
  const onChatScroll = (e) => {
    const a = e.currentTarget
    stickRef.current = a.scrollHeight - a.scrollTop - a.clientHeight < 80
  }
  useEffect(() => {
    const area = chatAreaRef.current
    if (area && stickRef.current) area.scrollTop = area.scrollHeight
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
    if (isSending || !activeSession) return null
    stickRef.current = true  // gửi tin mới thì luôn theo dõi câu trả lời
    return onSendMessage(text, files)
  }

  const handleSend = async () => {
    if (isSending || !activeSession) return
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
  const messages = activeSession?.messages || []
  const assessmentMessages = activeSession?.profile_summary?.length === 0 ? [] : messages.slice(activeSession?._assessmentStart || 0)
  const assessmentMessage = [...assessmentMessages].reverse().find(msg => msg.sender === 'bot' && msg.data?.consensus && !msg.is_error)
  const latestReply = [...messages.slice(activeSession?._assessmentStart || 0)].reverse().find(msg => msg.sender === 'bot' && !msg._loading)

  const liveMsg = messages.find(m => m._loading)
  const lastBot = [...messages].reverse().find(m => m.sender === 'bot' && !m._loading && m.data?.transcript)
  const running = !!liveMsg
  const flowSteps = liveMsg ? liveMsg._steps : lastBot?.data?.transcript || []
  const flowGraph = liveMsg ? liveMsg._graph : lastBot?.data?.graph
  const turns = messages.filter(m => m.sender === 'user').length
  const debateRounds = lastBot?.data?.consensus?.debate_rounds
  const ask = (q) => { setModal(null); sendText(q) }
  const completeProfile = async (fields) => {
    const { profile_summary: summary, rejected } = await saveProfile(activeSession.id, fields)
    onProfileSaved(summary)
    setModal(null)
    showToast(rejected.length ? `Đã lưu hồ sơ (bỏ ${rejected.length} giá trị không hợp lệ).` : 'Đã lưu hồ sơ. Bạn có thể hỏi trợ lý về hồ sơ này.')
  }

  return (
    <main className="main-content consultation-workspace">
      {toastMsg && <div className="nb-toast" role="status">{toastMsg}</div>}

      <div className="nb-view nb-consult">
        <ProfileNotebook profileSummary={activeSession?.profile_summary} disabled={isSending || !activeSession}
          onEdit={() => setModal('form')} onShowResult={assessmentMessage ? () => setModal('result') : null} onRestart={onRestart} />

        <div className="nb-center">
        <section className="chat-panel" aria-labelledby="assistant-heading">
          <div className="chat-panel-heading nb-chat-head">
            <div className="nb-head-top">
              <div className="nb-brand"><img src="/logo.png" alt="" /><h1>Diabetes <mark>MultiAgent</mark></h1></div>
              <span className="nb-crumb">/ Phiên #{activeSession?.id ? String(activeSession.id).slice(0, 6) : '…'} · lượt {turns}</span>
              <div className="nb-head-actions">
                {llmLabel && <span className="system-badge" title={`${systemInfo.agents.length} tác tử · KG: ${systemInfo.knowledge_graph} · Embedding: ${systemInfo.embedding}`}><Cpu size={14} /> {llmLabel}</span>}
                <button className="icon-btn" onClick={() => setModal('form')} disabled={!activeSession} aria-label="Nhập hồ sơ" title="Nhập hồ sơ"><ClipboardList size={18} /></button>
                <button className="icon-btn" onClick={onRestart} disabled={isSending || !activeSession} aria-label="Bắt đầu lại" title="Xóa dữ liệu và bắt đầu lượt tư vấn mới"><RotateCcw size={18} /></button>
                <button className="icon-btn" onClick={onToggleTheme} aria-label={isDarkMode ? 'Chuyển giao diện sáng' : 'Chuyển giao diện tối'} title={isDarkMode ? 'Giao diện sáng' : 'Giao diện tối'}>
                  {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
                </button>
                <button className="icon-btn" onClick={onOpenHelp} aria-label="Giới thiệu" title="Giới thiệu"><HelpCircle size={18} /></button>
              </div>
            </div>
            <div className="nb-title-row">
              <h2 id="assistant-heading">{lastBot?.data?.consensus ? 'Đánh giá nguy cơ đái tháo đường' : lastBot?.data?.intent === 'KNOWLEDGE_QUESTION' ? 'Hỏi đáp kiến thức' : 'Hỏi trợ lý sức khỏe'}</h2>
              <div className="nb-chips">
                {lastBot?._thinkSeconds != null && <span className="nb-chip c-orange">{lastBot._thinkSeconds} s</span>}
                {systemInfo && <span className="nb-chip c-green">{systemInfo.agents.length} tác tử</span>}
                {debateRounds != null && <span className="nb-chip c-violet">{debateRounds} vòng tranh biện</span>}
              </div>
            </div>
            <p><span className="online-dot" aria-hidden="true" />Dữ liệu → ML/DL ‖ KB → RAG, sau đó hội đồng phản biện trước khi kết luận.</p>
          </div>

      <div className="chat-area" ref={chatAreaRef} onScroll={onChatScroll} role="log" aria-label="Cuộc trò chuyện với trợ lý AI" aria-live="polite">
        <div className="chat-container">
          {(!activeSession || !activeSession.messages || activeSession.messages.length === 0) ? (
            <div className="welcome-screen">
              <div className="welcome-bubble">
                <img src="/logo.png" alt="" className="welcome-logo" />
                <div>
                  <h3 className="welcome-heading">Xin chào! Tôi có thể giúp gì cho bạn?</h3>
                  <p className="welcome-sub">Hãy hỏi tôi bất cứ điều gì về đái tháo đường, hoặc gửi ảnh phiếu xét nghiệm để tôi đọc giúp. Bạn cũng có thể bấm “Nhập hồ sơ” để điền chỉ số theo từng bước.</p>
                </div>
              </div>
              <p className="suggestions-title">Câu hỏi thường gặp</p>
              <div className="chip-row">
                {SUGGESTIONS.map(s => (
                  <button key={s.label} className="chip" disabled={isSending || !activeSession} onClick={() => sendText(s.text)} title={s.text}>
                    {s.label}
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
                              title="Hữu ích" aria-label="Đánh dấu hữu ích"
                            >
                              <ThumbsUp size={16} fill={isLiked ? 'currentColor' : 'none'} />
                            </button>
                            <button
                              className={`action-btn${isDisliked ? ' disliked' : ''}`}
                              onClick={() => handleFeedback(msg, 'disliked')}
                              disabled={!!feedbackLoading[msg.id]}
                              title="Không hữu ích" aria-label="Đánh dấu không hữu ích"
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
                        title="Copy" aria-label="Sao chép tin nhắn"
                      >
                        <Copy size={16} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

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
                  <button onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== idx))} title="Bỏ file" aria-label={`Bỏ file ${file.name}`}>
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
          {showPlusMenu && (
            <div className="plus-menu" id="chat-attachments-menu" ref={menuRef}>
              <button className="menu-item" onClick={() => fileInputRef.current.click()}>
                <Paperclip size={18} />
                <span>Tải bệnh án / phiếu xét nghiệm (PDF, ảnh, DOCX, CSV)</span>
              </button>
              <button className="menu-item" onClick={() => {
                setShowCamera(true);
                setShowPlusMenu(false);
              }}>
                <Camera size={18} />
                <span>Chụp ảnh phiếu xét nghiệm</span>
              </button>
              <button className="menu-item" onClick={() => {
                setShowPlusMenu(false)
                onOpenSettings()
              }}>
                <Settings size={18} />
                <span>Cài đặt</span>
              </button>
            </div>
          )}

          <div className="input-wrapper">
            <button
              className="icon-btn"
              onClick={() => setShowPlusMenu(!showPlusMenu)}
              title="Đính kèm bệnh án" aria-label="Đính kèm bệnh án" aria-expanded={showPlusMenu} aria-controls="chat-attachments-menu"
              disabled={isSending || !activeSession}
            >
              <Plus size={20} />
            </button>
            <input
              type="text"
              placeholder="Nhập câu hỏi của bạn…"
              aria-label="Tin nhắn cho trợ lý AI"
              disabled={!activeSession}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.nativeEvent.isComposing && !isSending && (inputValue.trim() || selectedFiles.length > 0)) {
                  handleSend();
                }
              }}
            />
            <button className={`icon-btn ${isListening ? 'recording' : ''}`} onClick={toggleListening}
              disabled={isTranscribing || isSending || !activeSession}
              aria-label={isListening ? 'Dừng ghi âm' : 'Ghi âm giọng nói'}
              title={isListening ? 'Dừng ghi âm' : isTranscribing ? 'Đang xử lý...' : 'Ghi âm giọng nói'}
            >
              {isTranscribing
                ? <Hourglass size={20} className="spin-hourglass" />
                : isListening
                ? <Circle size={20} fill="currentColor" className="record-icon" />
                : <Mic size={20} />}
            </button>
            <button
              className="icon-btn chat-send-button"
              disabled={isSending || !activeSession || (!inputValue.trim() && selectedFiles.length === 0)}
              aria-label="Gửi tin nhắn" title="Gửi tin nhắn"
              onClick={handleSend}
            >
              <Send size={20} />
            </button>
          </div>

          <div className="disclaimer">
            Chỉ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ. Trường hợp khẩn cấp, gọi 115.
          </div>
        </div>
      </div>
        </section>
        <PlanTrack steps={flowSteps} live={running} />
        </div>

        <AgentFlow steps={flowSteps} live={running} graph={flowGraph} onOpenKb={() => setModal('kb')} onOpenAgents={() => setModal('agents')} />
      </div>

      {(modal === 'form' || modal === 'result') && activeSession && (
        <NbModal title={modal === 'form' ? 'Hồ sơ bệnh nhân' : 'Kết quả đánh giá'} icon={modal === 'form' ? ClipboardList : Gauge}
          subtitle={modal === 'form' ? 'Khoảng 2 phút · 3 bước · bấm Hoàn thành để lưu, không bắt buộc có xét nghiệm' : 'Kết luận của hội đồng tác tử cho hồ sơ hiện tại'} onClose={closeModal}
          actions={modal === 'form' && <button className="btn-secondary nb-skip" onClick={closeModal}>Bỏ qua</button>}>
          <PredictionPanel key={modal} initialView={modal}
            profileSummary={activeSession.profile_summary} isSending={isSending} disabled={!activeSession}
            onPredict={text => { setModal(null); return sendText(text) }} onAsk={ask} onComplete={completeProfile}
            assessmentMessage={assessmentMessage} latestReply={latestReply} />
        </NbModal>
      )}
      {modal === 'kb' && <NbModal wide title="Tri thức KB" icon={Database} subtitle="PDF QĐ 5481/QĐ-BYT + Vinmec + Long Châu + ADA 2024 → Qdrant + Neo4j" onClose={closeModal}><KnowledgeView /></NbModal>}
      {modal === 'agents' && <NbModal wide title="Hệ tác tử" icon={Bot} subtitle={`${systemInfo?.agents.length ?? 9} tác tử trao đổi qua MessageBus (FIPA-ACL)`} onClose={closeModal}><AgentsView systemInfo={systemInfo} /></NbModal>}
    </main>
  )
}

export default MainChat
