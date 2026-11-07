import { useEffect, useRef, useState } from 'react';
import { X, Camera } from 'lucide-react';

function CameraModal({ onClose, onCapture }) {
  const videoRef = useRef(null);
  const [error, setError] = useState('');
  
  useEffect(() => {
    let stream = null;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Lỗi khi mở camera:", err);
        setError("Không thể mở camera. Vui lòng kiểm tra lại quyền truy cập.");
      }
    };
    
    startCamera();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const handleCapture = () => {
    if (!videoRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' });
        onCapture(file);
      }
    }, 'image/jpeg', 0.9);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{ maxWidth: '500px', width: '90%' }}>
        <button className="modal-close" onClick={onClose}>
          <X size={24} />
        </button>
        <h2 style={{ marginBottom: '16px', textAlign: 'center' }}>Chụp ảnh</h2>
        
        {error ? (
          <div style={{ padding: '20px', color: '#ef4444', textAlign: 'center' }}>{error}</div>
        ) : (
          <div style={{ position: 'relative', width: '100%', height: '400px', backgroundColor: '#000', borderRadius: '12px', overflow: 'hidden' }}>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          </div>
        )}
        
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
          <button 
            onClick={handleCapture}
            disabled={!!error}
            style={{
              backgroundColor: 'var(--primary-color)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '64px',
              height: '64px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: error ? 'not-allowed' : 'pointer',
              transition: 'transform 0.2s'
            }}
            onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.9)'}
            onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
          >
            <Camera size={32} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default CameraModal;
