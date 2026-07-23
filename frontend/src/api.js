import axios from 'axios';

const API_BASE_URL = 'http://localhost:10000';
// const API_BASE_URL = 'https://skin-pigmentation-detection.onrender.com';

// ============================================
// FIX: Add timeout and retry logic
// ============================================
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for cold starts
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// ============================================
// FIX: Health check to warm up backend
// ============================================
export const checkBackendHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 10000 });
    return response.data.status === 'ok';
  } catch (error) {
    console.warn('Backend health check failed:', error.message);
    return false;
  }
};

// ============================================
// FIX: Compress image before upload if > 1MB
// ============================================
const compressImage = async (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;
        
        // Resize if too large
        const maxDim = 1024;
        if (width > maxDim || height > maxDim) {
          if (width > height) {
            height = (height / width) * maxDim;
            width = maxDim;
          } else {
            width = (width / height) * maxDim;
            height = maxDim;
          }
        }
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        
        canvas.toBlob(
          (blob) => {
            resolve(new File([blob], file.name, { type: 'image/jpeg' }));
          },
          'image/jpeg',
          0.85
        );
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
};

export const predictPigmentation = async (clinicalImage, manualData = {}) => {
  const formData = new FormData();
  
  if (clinicalImage) {
    // FIX: Compress image if larger than 1MB
    let imageToUpload = clinicalImage;
    if (clinicalImage.size > 1024 * 1024) {
      console.log('Compressing image...');
      imageToUpload = await compressImage(clinicalImage);
    }
    formData.append('clinical_image', imageToUpload);
  }
  
  // Add manual data fields if provided
  if (manualData.age) formData.append('age', manualData.age);
  if (manualData.affected_area) formData.append('affected_area', manualData.affected_area);
  if (manualData.pigmentation_intensity) formData.append('pigmentation_intensity', manualData.pigmentation_intensity);
  if (manualData.duration) formData.append('duration', manualData.duration);
  if (manualData.progression) formData.append('progression', manualData.progression);
  if (manualData.itching) formData.append('itching', manualData.itching);
  if (manualData.burning) formData.append('burning', manualData.burning);
  if (manualData.pain) formData.append('pain', manualData.pain);
  if (manualData.sun_exposure) formData.append('sun_exposure', manualData.sun_exposure);
  if (manualData.sunscreen_use) formData.append('sunscreen_use', manualData.sunscreen_use);
  if (manualData.user_concern) formData.append('user_concern', manualData.user_concern);

  const response = await api.post('/predict', formData);
  return response.data;
};