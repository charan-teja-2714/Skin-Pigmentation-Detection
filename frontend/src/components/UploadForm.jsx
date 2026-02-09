import React, { useState, useEffect } from 'react';
import { predictPigmentation, checkBackendHealth } from '../api';
import CameraCapture from './CameraCapture';

const UploadForm = () => {
  const [clinicalImage, setClinicalImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  // ============================================
  // FIX: Add backend warming state
  // ============================================
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'ready', 'warming'
  
  // Manual input fields
  const [manualData, setManualData] = useState({
    age: '',
    affected_area: '',
    pigmentation_intensity: '',
    duration: '',
    progression: '',
    itching: '',
    burning: '',
    pain: '',
    sun_exposure: '',
    sunscreen_use: '',
    user_concern: ''
  });

  // ============================================
  // FIX: Pre-ping backend on component mount
  // ============================================
  useEffect(() => {
    const warmUpBackend = async () => {
      setBackendStatus('checking');
      const isHealthy = await checkBackendHealth();
      if (isHealthy) {
        setBackendStatus('ready');
      } else {
        setBackendStatus('warming');
        // Retry after 5 seconds
        setTimeout(async () => {
          const retry = await checkBackendHealth();
          setBackendStatus(retry ? 'ready' : 'warming');
        }, 5000);
      }
    };
    warmUpBackend();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const hasImage = clinicalImage !== null;
    const hasManualInputs = Object.values(manualData).some(value => value !== '');
    
    if (!hasImage && !hasManualInputs) {
      setError('Please provide either an image or fill in the assessment fields');
      return;
    }

    // ============================================
    // FIX: Show warming message if backend not ready
    // ============================================
    if (backendStatus === 'warming') {
      setError('Backend is warming up. Please wait a moment and try again.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const cleanManualData = {};
      Object.keys(manualData).forEach(key => {
        if (manualData[key] !== '') {
          cleanManualData[key] = manualData[key];
        }
      });
      
      const prediction = await predictPigmentation(clinicalImage, cleanManualData);
      setResult(prediction);
      setBackendStatus('ready'); // Mark as ready after successful request
    } catch (err) {
      // ============================================
      // FIX: Better error handling for timeouts
      // ============================================
      if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
        setError('Request timed out. The backend may be warming up. Please try again in a moment.');
        setBackendStatus('warming');
      } else {
        const errorMessage = err.response?.data?.detail || err.message || 'Prediction failed';
        setError(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
      }
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

  const handleManualInputChange = (field, value) => {
    setManualData(prev => ({ ...prev, [field]: value }));
  };

  const openCamera = () => {
    setShowCamera(true);
  };

  return (
    <div className="upload-form">
      <h2>Skin Pigmentation Assessment</h2>
      
      {/* ============================================ */}
      {/* FIX: Show backend status indicator */}
      {/* ============================================ */}
      {backendStatus === 'checking' && (
        <div className="info-banner" style={{backgroundColor: '#fff3cd', padding: '10px', borderRadius: '5px', marginBottom: '15px'}}>
          ⏳ Checking backend status...
        </div>
      )}
      {backendStatus === 'warming' && (
        <div className="warning-banner" style={{backgroundColor: '#fff3cd', padding: '10px', borderRadius: '5px', marginBottom: '15px'}}>
          🔥 Backend is warming up. This may take 30-60 seconds on first request. Please wait...
        </div>
      )}
      {backendStatus === 'ready' && (
        <div className="success-banner" style={{backgroundColor: '#d4edda', padding: '10px', borderRadius: '5px', marginBottom: '15px'}}>
          ✅ Backend is ready
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="form-container">
          {/* Left Side - Manual Assessment */}
          <div className="assessment-section">
            <h3>📋 Clinical Assessment</h3>

            <div className="form-group">
              <label>Age:</label>
              <input
                type="number"
                min="1"
                max="120"
                placeholder="Enter your age"
                value={manualData.age}
                onChange={(e) => handleManualInputChange('age', e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Affected Area (Extent):</label>
              <select value={manualData.affected_area} onChange={(e) => handleManualInputChange('affected_area', e.target.value)}>
                <option value="">Select extent</option>
                <option value="small_spots">Small Spots</option>
                <option value="several_small_spots">Several Small Spots</option>
                <option value="large_patches">Large Patches</option>
                <option value="most_of_area">Most of Area</option>
              </select>
            </div>

            <div className="form-group">
              <label>Pigmentation Intensity:</label>
              <select value={manualData.pigmentation_intensity} onChange={(e) => handleManualInputChange('pigmentation_intensity', e.target.value)}>
                <option value="">Select intensity</option>
                <option value="light">Light</option>
                <option value="medium">Medium</option>
                <option value="dark">Dark</option>
              </select>
            </div>

            <div className="form-group">
              <label>Duration of Pigmentation:</label>
              <select value={manualData.duration} onChange={(e) => handleManualInputChange('duration', e.target.value)}>
                <option value="">Select duration</option>
                <option value="less_than_1_month">Less than 1 month</option>
                <option value="one_to_six_months">1-6 months</option>
                <option value="more_than_six_months">More than 6 months</option>
                <option value="several_years">Several years</option>
              </select>
            </div>

            <div className="form-group">
              <label>Progression Over Time:</label>
              <select value={manualData.progression} onChange={(e) => handleManualInputChange('progression', e.target.value)}>
                <option value="">Select progression</option>
                <option value="stable">Stable</option>
                <option value="slowly_increasing">Slowly Increasing</option>
                <option value="rapidly_increasing">Rapidly Increasing</option>
              </select>
            </div>

            <div className="symptoms-group">
              <h4>Symptoms</h4>
              <div className="checkbox-group">
                <div className="form-group">
                  <label>Itching:</label>
                  <select value={manualData.itching} onChange={(e) => handleManualInputChange('itching', e.target.value)}>
                    <option value="">Select</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Burning Sensation:</label>
                  <select value={manualData.burning} onChange={(e) => handleManualInputChange('burning', e.target.value)}>
                    <option value="">Select</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Pain or Discomfort:</label>
                  <select value={manualData.pain} onChange={(e) => handleManualInputChange('pain', e.target.value)}>
                    <option value="">Select</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="contextual-group">
              <h4>Contextual Information</h4>
              <div className="form-group">
                <label>Sun Exposure Level:</label>
                <select value={manualData.sun_exposure} onChange={(e) => handleManualInputChange('sun_exposure', e.target.value)}>
                  <option value="">Select exposure</option>
                  <option value="low">Low</option>
                  <option value="moderate">Moderate</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div className="form-group">
                <label>Sunscreen Usage:</label>
                <select value={manualData.sunscreen_use} onChange={(e) => handleManualInputChange('sunscreen_use', e.target.value)}>
                  <option value="">Select usage</option>
                  <option value="regularly">Regularly</option>
                  <option value="occasionally">Occasionally</option>
                  <option value="never">Never</option>
                </select>
              </div>
              <div className="form-group">
                <label>Your Concern Level:</label>
                <select value={manualData.user_concern} onChange={(e) => handleManualInputChange('user_concern', e.target.value)}>
                  <option value="">Select concern</option>
                  <option value="not_concerned">Not Concerned</option>
                  <option value="somewhat_concerned">Somewhat Concerned</option>
                  <option value="very_concerned">Very Concerned</option>
                </select>
              </div>
            </div>
          </div>

          {/* Right Side - Image Upload */}
          <div className="image-section">
            <h3>📷 Image Analysis</h3>
            <div className="form-group">
              <label htmlFor="clinical">Upload Skin Image:</label>
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
          </div>
        </div>

        <div className="submit-section">
          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Pigmentation'}
          </button>
        </div>
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
                  <p><strong>Input Method:</strong> {result.features.input_method}</p>
                  {Object.entries(result.features).map(([key, value]) => {
                    if (key !== 'input_method' && value !== undefined && value !== null) {
                      const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                      const displayValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
                      return <p key={key}><strong>{displayKey}:</strong> {displayValue}</p>;
                    }
                    return null;
                  })}
                </div>
              )}
            </div>

            {result.advisory && (
              <div className="advisory-card">
                <h4>🩺 Medical Advisory</h4>
                <div className="advisory-content">
                  {result.advisory.split('\n').map((line, index) => {
                    // Clean the line thoroughly
                    let cleanLine = line.trim()
                      .replace(/^\*+\s*/, '') // Remove asterisks at start
                      .replace(/\*+$/, '') // Remove asterisks at end
                      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove **bold**
                      .replace(/\*(.*?)\*/g, '$1') // Remove *italic*
                      .replace(/^[*-]\s*/, '') // Remove bullet points
                      .replace(/^#+\s*/, '') // Remove headers
                      .replace(/^\d+\.\s*/, '') // Remove numbered lists
                      .replace(/&#39;/g, "'") // Fix HTML entities
                      .replace(/&quot;/g, '"')
                      .replace(/&amp;/g, '&');
                    
                    // Skip empty lines
                    if (!cleanLine) return null;
                    
                    // Check content type
                    const isDisclaimer = cleanLine.toLowerCase().includes('disclaimer') || 
                                       cleanLine.toLowerCase().includes('not a medical diagnosis') ||
                                       cleanLine.toLowerCase().includes('consult a qualified healthcare');
                    
                    const isHeading = (cleanLine.includes(':') && cleanLine.length < 80) ||
                                    cleanLine.toLowerCase().includes('understanding') ||
                                    cleanLine.toLowerCase().includes('interpreting') ||
                                    cleanLine.toLowerCase().includes('recommendations');
                    
                    const isRecommendation = /^(sun protection|adequate hydration|gentle skincare|avoid|wear|drink|use|seek|apply)/i.test(cleanLine);
                    
                    return (
                      <div key={index} className={`advisory-item ${
                        isDisclaimer ? 'disclaimer' : 
                        isHeading ? 'advisory-heading' : 
                        isRecommendation ? 'recommendation' : 'advisory-text'
                      }`}>
                        {isRecommendation && <span className="bullet">•</span>}
                        <span>{cleanLine}</span>
                      </div>
                    );
                  }).filter(Boolean)}
                </div>
              </div>
            )}
          </div>
          
          {/* {result.masks && (
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
          )} */}
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