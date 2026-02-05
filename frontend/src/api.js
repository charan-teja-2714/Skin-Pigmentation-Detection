import axios from 'axios';

// const API_BASE_URL = 'http://localhost:8000';
const API_BASE_URL = 'https://skin-pigmentation-detection.onrender.com';


export const predictPigmentation = async (clinicalImage, manualData = {}) => {
  const formData = new FormData();
  
  if (clinicalImage) {
    formData.append('clinical_image', clinicalImage);
  }
  
  // Add manual data fields if provided
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

  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};