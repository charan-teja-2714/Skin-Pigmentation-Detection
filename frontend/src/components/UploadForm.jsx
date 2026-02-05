import React, { useState } from 'react';
import { predictPigmentation } from '../api';
import CameraCapture from './CameraCapture';

const UploadForm = () => {
  const [clinicalImage, setClinicalImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showCamera, setShowCamera] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!clinicalImage) {
      setError('Clinical image is required');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictPigmentation(clinicalImage);
      setResult(prediction);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCameraCapture = (file) => {
    setClinicalImage(file);
    setImagePreview(URL.createObjectURL(file));
    setShowCamera(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setClinicalImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const openCamera = () => {
    setShowCamera(true);
  };

  return (
    <div className="upload-form">
      <h2>Skin Pigmentation Detection</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="clinical">Upload Skin Image (Required):</label>
          <div className="input-group">
            <input
              type="file"
              id="clinical"
              accept="image/*"
              onChange={handleFileChange}
            />
            <button type="button" onClick={openCamera} className="camera-btn">
              📷 Camera
            </button>
          </div>
          {clinicalImage && <p className="file-name">Selected: {clinicalImage.name}</p>}
        </div>

        {imagePreview && (
          <div className="image-preview">
            <h4>Selected Image:</h4>
            <img src={imagePreview} alt="Selected skin image" className="preview-image" />
          </div>
        )}

        <button type="submit" disabled={loading || !clinicalImage}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {error && (
        <div className="error">
          <h3>Error:</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="result">
          <h3>Analysis Result:</h3>
          <div className="result-grid">
            <div className="metrics-card">
              <h4>📊 Analysis Metrics</h4>
              <p><strong>Score:</strong> {result.score}</p>
              <p><strong>Severity:</strong> <span className={`severity-${result.severity.toLowerCase()}`}>{result.severity}</span></p>
              {result.features && (
                <div className="features">
                  <p><strong>Pigmented Area:</strong> {result.features.pigmented_area_pct}%</p>
                  <p><strong>Contrast:</strong> {result.features.contrast}</p>
                </div>
              )}
            </div>

            {result.advisory && (
              <div className="advisory-card">
                <h4>🩺 Medical Advisory</h4>
                <div className="advisory-content">
                  {result.advisory.split('\n').filter(line => line.trim()).map((line, index) => {
                    const isDisclaimer = line.toLowerCase().includes('disclaimer') || 
                                       line.toLowerCase().includes('not a medical diagnosis') ||
                                       line.toLowerCase().includes('consult');
                    const isHeading = line.includes(':') && !line.includes('•') && line.length < 100;
                    const isBullet = line.trim().startsWith('*') || line.trim().startsWith('-');
                    
                    // Clean up the line
                    let cleanLine = line;
                    if (isBullet) {
                      cleanLine = line.replace(/^\s*[*-]\s*/, ''); // Remove bullet markers
                    }
                    
                    // Handle bold text **text**
                    const parts = cleanLine.split(/\*\*(.*?)\*\*/);
                    
                    return (
                      <p key={index} className={`
                        ${isDisclaimer ? 'disclaimer' : ''}
                        ${isHeading ? 'advisory-heading' : ''}
                        ${isBullet ? 'bullet-point' : ''}
                      `.trim()}>
                        {parts.map((part, i) => 
                          i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                        )}
                      </p>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          
          {result.masks && (
            <div className="masks-section">
              <h4>🔍 Analysis Visualization</h4>
              <div className="masks-container">
                <div className="mask-item">
                  <h5>Skin Detection</h5>
                  <img src={result.masks.skin_mask} alt="Skin mask" className="mask-image" />
                </div>
                <div className="mask-item">
                  <h5>Pigmentation Detection</h5>
                  <img src={result.masks.pigmentation_mask} alt="Pigmentation mask" className="mask-image" />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
};

export default UploadForm;