import runpod
import os
import base64
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        # ضبط الإعدادات v1 لضمان التوافق التام مع الموديلات المستقرة في منطقتك
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # الترجمة والتحسين
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Executing: {final_optimized_prompt}")

        # طلب التوليد مع تحديد فلاتر الأمان لتكون أقل تقييداً (لتجنب التعليق)
        # نستخدم gemini-2.5-flash كونه الأنسب لحسابك المدفوع حالياً
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "ACT AS AN IMAGE GENERATOR. RETURN THE IMAGE DATA IMMEDIATELY.",
                f"A cinematic high-quality photo: {final_optimized_prompt}"
            ]
        )

        # التحقق الآمن من الرد لمنع التعليق المستمر
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            raise Exception("No response candidates. The model might be taking too long or blocked the content.")

        image_bytes = None
        # استخراج البيانات بشكل مباشر وسريع
        candidate = response.candidates[0]
        if hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break

        if image_bytes:
            print("--- SUCCESS: Image Captured! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            raise Exception("Model responded but provided no image data. Try a simpler prompt.")

    except Exception as e:
        print(f"--- [TIMEOUT OR ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
