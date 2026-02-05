import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMAdvisor:
    def __init__(self):
        # Initialize Groq client using LangChain
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                self.client = ChatGroq(
                    groq_api_key=api_key,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.2,
                    max_tokens=200
                )
            except Exception as e:
                print(f"Groq client initialization error: {e}")
                self.client = None
        else:
            self.client = None

        self.system_prompt = """You are a skin health advisory assistant integrated into a clinical decision-support system.

CRITICAL RULES (DO NOT VIOLATE):
- You do NOT perform medical diagnosis.
- You do NOT identify or name diseases.
- You do NOT change, override, or recalculate severity scores.
- You do NOT generate urgency language such as "immediately" or "urgent".
- You only provide general guidance and precautions.
- You must always include a clear medical disclaimer.

SYSTEM CONTEXT:
Severity results are generated upstream using image-based machine learning models.
Severity scores and labels are already finalized and authoritative.

YOUR TASK:
Generate a calm, supportive, and user-friendly advisory response that includes:

1. A brief explanation of what the given severity level means in simple, non-clinical language.
2. General guidance on whether consulting a dermatologist may be beneficial, phrased as a suggestion only.
3. Practical, safe precautions for managing skin pigmentation:
   - Sun protection
   - Hydration
   - Gentle skincare
   - Avoiding known irritants
4. A clear disclaimer stating this is not a medical diagnosis.

STYLE AND TONE REQUIREMENTS:
- Be reassuring and neutral.
- Do not alarm the user.
- Do not speculate about diseases.
- Keep the response concise (100–180 words).
- Use structured formatting.

DISCLAIMER (MANDATORY):
End every response with: "This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice."

"""


    def get_llm_advice(self, severity_score, severity_level, area_pct, contrast):
        """Get LLM-generated advisory text based on severity analysis"""

        # Handle missing API key gracefully
        if not self.client:
            return self._fallback_advice(severity_level)

        try:
            # Convert contrast to descriptive term
            contrast_desc = self._get_contrast_description(contrast)

            user_prompt = f"""Based on skin pigmentation analysis results:
- Severity Level: {severity_level}
- Severity Score: {severity_score:.2f}
- Pigmented Area: {area_pct:.1f}%
- Contrast: {contrast_desc}

Provide general skin health guidance following the clinical decision-support system guidelines."""

            messages = [
                ("system", self.system_prompt),
                ("human", user_prompt)
            ]
            
            response = self.client.invoke(messages)
            return response.content.strip()

        except Exception as e:
            print(f"LLM Advisory Error: {e}")
            return self._fallback_advice(severity_level)

    def _get_contrast_description(self, contrast_value):
        """Convert numerical contrast to descriptive term"""
        if contrast_value < 0.2:
            return "low"
        elif contrast_value < 0.4:
            return "medium"
        else:
            return "high"

    def _fallback_advice(self, severity_level):
        """Fallback advice when LLM is unavailable - follows clinical guidelines"""
        advice_map = {
            "Mild": """The analysis indicates mild pigmentation changes in your skin. This suggests minimal variation in skin tone that may be within normal ranges.

Consulting a dermatologist may be beneficial for professional evaluation and personalized guidance.

General precautions that may help:
• Use broad-spectrum sunscreen daily (SPF 30+)
• Stay hydrated by drinking adequate water
• Use gentle, fragrance-free skincare products
• Avoid harsh scrubbing or irritating products

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice.""",
            
            "Moderate": """The analysis shows moderate pigmentation patterns in your skin. This indicates more noticeable variations in skin tone that may benefit from professional attention.

Consulting a dermatologist would be advisable for proper evaluation and guidance on management options.

General precautions that may help:
• Apply broad-spectrum sunscreen daily and reapply frequently
• Maintain good hydration
• Use gentle, non-irritating skincare products
• Avoid excessive sun exposure, especially during peak hours

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice.""",
            
            "Severe": """The analysis indicates more pronounced pigmentation changes in your skin. This suggests significant variations in skin tone that would benefit from professional evaluation.

Consulting a dermatologist is recommended for comprehensive assessment and appropriate guidance.

General precautions that may help:
• Use broad-spectrum sunscreen consistently (SPF 30+)
• Seek shade and limit sun exposure
• Maintain gentle skincare routines
• Stay well-hydrated
• Avoid known skin irritants

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice."""
        }

        return advice_map.get(
            severity_level,
            "Please consult a qualified healthcare professional for proper skin assessment. This information is for general guidance only and is not a medical diagnosis.",
        )
