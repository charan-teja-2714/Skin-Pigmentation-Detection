import React, { useState } from 'react';
import { predictPigmentation } from '../api';

const UploadForm = () => {
  const [clinicalImage, setClinicalImage] = useState(null);
  const [dermoscopyImage, setDermoscopyImage] = useState(null);
  const [multispectralImage, setMultispectralImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

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
      const prediction = await predictPigmentation(clinicalImage, dermoscopyImage, multispectralImage);
      setResult(prediction);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-form">
      <h2>Skin Pigmentation Detection</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="clinical">Clinical Image (Required):</label>
          <input
            type="file"
            id="clinical"
            accept="image/*"
            onChange={(e) => setClinicalImage(e.target.files[0])}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="dermoscopy">Dermoscopy Image (Optional):</label>
          <input
            type="file"
            id="dermoscopy"
            accept="image/*"
            onChange={(e) => setDermoscopyImage(e.target.files[0])}
          />
        </div>

        <div className="form-group">
          <label htmlFor="multispectral">Multispectral Image (Optional):</label>
          <input
            type="file"
            id="multispectral"
            accept="image/*"
            onChange={(e) => setMultispectralImage(e.target.files[0])}
          />
        </div>

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
          <p><strong>Score:</strong> {result.score}</p>
          <p><strong>Severity:</strong> {result.severity}</p>
        </div>
      )}
    </div>
  );
};

export default UploadForm;