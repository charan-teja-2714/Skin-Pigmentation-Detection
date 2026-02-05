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
                    max_tokens=1000
                )
            except Exception as e:
                print(f"Groq client initialization error: {e}")
                self.client = None
        else:
            self.client = None

        self.system_prompt = """You are a skin health advisory assistant integrated into a larger AI system.

IMPORTANT ROLE DEFINITION:
You do NOT predict severity.
You do NOT analyze images.
You do NOT override backend decisions.
You ONLY explain and provide general guidance based on results already computed.

SYSTEM GUARANTEES:
- Severity score and severity level (Mild / Moderate / Severe) are FINAL and authoritative.
- These results may be derived from image-based analysis, structured user inputs, or a combination.
- You must NOT speculate on how the severity was computed.
- You must treat all provided values as correct.

YOUR TASK:
Generate a user-friendly advisory response that includes:

1. A simple, non-clinical explanation of what the given severity level means.
2. A short interpretation that considers the provided context without making assumptions.
3. General, safe self-care recommendations for managing skin pigmentation:
   - Sun protection
   - Adequate hydration
   - Gentle skincare
   - Avoiding known irritants
4. A suggestion (not instruction) that consulting a dermatologist may be helpful, phrased calmly.
5. A clearly separated disclaimer stating this is not a medical diagnosis.

STYLE & SAFETY RULES:
- Do NOT name diseases or conditions.
- Do NOT use alarming or urgent language.
- Do NOT recommend medications.
- Do NOT give definitive medical conclusions.
- Keep the response supportive, neutral, and readable.
- Length: 120–200 words.
- Use clear paragraphs or numbered points.

MANDATORY DISCLAIMER:
"This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice."

Do NOT mention system architecture, models, fallback modes, or internal logic.
Do NOT ask follow-up questions.

"""


    def get_llm_advice(self, severity_score, severity_level, area_pct, contrast):
        """Get LLM-generated advisory text based on severity analysis"""

        # Handle missing API key gracefully
        if not self.client:
            return self._fallback_advice(severity_level)

        try:
            # Convert contrast to descriptive term
            contrast_desc = self._get_contrast_description(contrast)

            user_prompt = f"""Based on the computed analysis results:
- Severity Level: {severity_level}
- Severity Score: {severity_score:.2f}
- Pigmented Area: {area_pct:.1f}%
- Contrast: {contrast_desc}

Provide explanatory guidance based on these final results."""

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
        """Fallback advice when LLM is unavailable - explanatory guidance only"""
        advice_map = {
            "Mild": """Based on the computed analysis, your results indicate a mild severity level. This suggests relatively minor variations in skin pigmentation that are within a manageable range.

The analysis has identified some pigmentation patterns, but they appear to be at a lower intensity level. This is generally a more favorable outcome.

General self-care recommendations that may help:
• Apply broad-spectrum sunscreen daily (SPF 30 or higher)
• Stay well-hydrated throughout the day
• Use gentle, fragrance-free skincare products
• Avoid harsh scrubbing or abrasive treatments

Consulting a dermatologist may be helpful for personalized guidance and to discuss any concerns you might have.

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice.""",
            
            "Moderate": """Based on the computed analysis, your results indicate a moderate severity level. This suggests noticeable pigmentation variations that fall into a middle range of intensity.

The analysis has detected pigmentation patterns that are more pronounced than mild cases but not at the highest level. This indicates areas that may benefit from attention and care.

General self-care recommendations that may help:
• Use broad-spectrum sunscreen consistently and reapply regularly
• Maintain good hydration habits
• Choose gentle, non-irritating skincare products
• Limit sun exposure during peak hours (10 AM - 4 PM)

Consulting a dermatologist may be beneficial for professional evaluation and personalized management strategies.

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice.""",
            
            "Severe": """Based on the computed analysis, your results indicate a severe severity level. This suggests more significant pigmentation variations that represent the higher end of the intensity range.

The analysis has identified pigmentation patterns that are more pronounced and extensive. While this indicates areas that warrant attention, remember that these are computational results that benefit from professional interpretation.

General self-care recommendations that may help:
• Use broad-spectrum sunscreen diligently (SPF 30+)
• Seek shade and minimize direct sun exposure
• Maintain consistent, gentle skincare routines
• Stay well-hydrated and avoid known skin irritants

Consulting a dermatologist would be helpful for comprehensive evaluation and to discuss appropriate management options.

This information is for general guidance only and is not a medical diagnosis. Please consult a qualified healthcare professional for personalized medical advice."""
        }

        return advice_map.get(
            severity_level,
            "Please consult a qualified healthcare professional for proper evaluation of your skin. This information is for general guidance only and is not a medical diagnosis.",
        )
