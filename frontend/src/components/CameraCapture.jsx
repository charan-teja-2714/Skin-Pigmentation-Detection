import React, { useRef, useState, useEffect } from 'react';

const CameraCapture = ({ onCapture, onClose }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [capturedFile, setCapturedFile] = useState(null);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsStreaming(false);
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      // Get image data URL for preview
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      setCapturedImage(imageDataUrl);
      
      // Create file for submission
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        setCapturedFile(file);
        stopCamera();
      }, 'image/jpeg', 0.8);
    }
  };

  const confirmCapture = () => {
    if (capturedFile) {
      onCapture(capturedFile);
      stopCamera();  // Ensure camera is stopped
      onClose();
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    setCapturedFile(null);
    startCamera();
  };

  const handleClose = () => {
    stopCamera();  // Always stop camera when closing
    onClose();
  };

  useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="camera-modal">
      <div className="camera-content">
        <h3>Capture Image</h3>
        
        <div className="camera-container">
          {!capturedImage ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />
          ) : (
            <img 
              src={capturedImage} 
              alt="Captured" 
              className="captured-image"
            />
          )}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
        
        <div className="camera-controls">
          {!capturedImage ? (
            <>
              <button 
                onClick={captureImage} 
                disabled={!isStreaming}
                className="capture-btn"
              >
                📷 Capture
              </button>
              <button 
                onClick={() => { 
                  stopCamera(); 
                  handleClose(); 
                }} 
                className="cancel-btn"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <button 
                onClick={confirmCapture}
                className="capture-btn"
              >
                ✓ Use This Photo
              </button>
              <button 
                onClick={retakePhoto}
                className="retake-btn"
              >
                🔄 Retake
              </button>
              <button 
                onClick={handleClose}
                className="cancel-btn"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraCapture;