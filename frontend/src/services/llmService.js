/**
 * Kết nối Hệ Thống Đa Tác Tử (Multi-Agent System) — backend FastAPI
 */

const parseError = async (response) => {
    try {
        const data = await response.json();
        return data.detail || `Lỗi HTTP ${response.status}`;
    } catch {
        return `Lỗi HTTP ${response.status}`;
    }
};

/**
 * Gửi một lượt chat. Có file thì gửi multipart tới /api/chat/upload, không thì JSON tới /api/chat.
 * Trả về { session, user_message, bot_message }.
 */
export const sendChat = async (sessionId, text, files = []) => {
    let response;
    if (files.length > 0) {
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        formData.append('text', text || '');
        if (sessionId) formData.append('session_id', sessionId);
        response = await fetch('/api/chat/upload', { method: 'POST', body: formData });
    } else {
        response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, text })
        });
    }
    if (!response.ok) throw new Error(await parseError(response));
    return response.json();
};

export const fetchSystemInfo = async () => {
    const response = await fetch('/api/system');
    if (!response.ok) throw new Error(await parseError(response));
    return response.json();
};

export const resetProfile = async (sessionId) => {
    const response = await fetch(`/api/sessions/${sessionId}/profile`, { method: 'DELETE' });
    if (!response.ok) throw new Error(await parseError(response));
};

/**
 * Chuyển giọng nói thành văn bản (Gemini)
 */
export const transcribeAudio = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "record.webm");
    const response = await fetch("/api/transcribe", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await parseError(response));
    const data = await response.json();
    return data.text || "";
};
