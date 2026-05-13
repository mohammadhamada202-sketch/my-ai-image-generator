import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # 1. إعداد العميل باستخدام المفتاح المدفوع (N5UI)
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 2. الترجمة عبر OpenAI
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Executing: {final_optimized_prompt}")

        # 3. طلب التوليد (بصيغة تجبر الموديل على إرسال بيانات الصورة فقط)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[
                "ACT AS AN IMAGE GENERATION ENGINE. RETURN ONLY THE INLINE_DATA BLOB. NO TEXT.",
                f"A cinematic 8k photo: {final_optimized_prompt}"
            ]
        )

        # 4. معالجة الرد ومنع خطأ 'NoneType'
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            raise Exception("No response candidates received from Gemini API.")

        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
            raise Exception("Response candidate has no content or parts.")

        image_bytes = None
        # فحص الأجزاء بشكل آمن لضمان عدم حدوث خطأ 'NoneType' object is not iterable
        parts = candidate.content.parts if candidate.content.parts is not None else []
        
        for part in parts:
            # التحقق من وجود بيانات الصورة (inline_data)
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                break
            elif hasattr(part, 'data') and part.data:
                image_bytes = part.data
                break

        if image_bytes:
            print("--- SUCCESS: Image data captured! ---")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            # إضافة ترويسة العرض لضمان ظهور الصورة على موقعك فوراً
            return f"data:image/png;base64,{encoded_image}"
        else:
            # محاولة قراءة النص التوضيحي إذا لم توجد صورة لفهم السبب
            debug_info = parts[0].text if len(parts) > 0 and hasattr(parts[0], 'text') else "No data found."
            raise Exception(f"Model returned OK but no image data. Info: {debug_info[:100]}")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
