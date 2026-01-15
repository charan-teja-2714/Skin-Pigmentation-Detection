import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const predictPigmentation = async (clinicalImage, dermoscopyImage, multispectralImage) => {
  const formData = new FormData();
  formData.append('clinical_image', clinicalImage);
  
  if (dermoscopyImage) {
    formData.append('dermoscopy_image', dermoscopyImage);
  }
  
  if (multispectralImage) {
    formData.append('multispectral_image', multispectralImage);
  }

  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};