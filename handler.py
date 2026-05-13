import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize
from dimensions_config import get_image_dimensions

def handler(job):
    try:
        # 1. إعداد المفتاح والعميل (v1 لضمان استقرار موديلات 2.5)
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 2. تحسين الترجمة عبر OpenAI
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Optimized Prompt: {final_optimized_prompt}")

        # 3. طلب التوليد (تم تعديل صياغة الطلب لإجبار الموديل على إخراج صورة)
        # استخدمنا gemini-2.5-flash لأنه الموديل النشط في حسابك
        print("Requesting image from gemini-2.5-flash...")
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[
                "ACT AS AN IMAGE GENERATION ENGINE. OUTPUT ONLY THE IMAGE DATA. NO TEXT. NO DESCRIPTIONS.",
                f"Generate a cinematic 8k photo of: {final_optimized_prompt}"
            ]
        )

        # 4. البحث المعمق عن بيانات الصورة داخل الرد 
        if not response or not response.candidates:
            raise Exception("No response from Gemini.")

        image_bytes = None
        for part in response.candidates[0].content.parts:
            # التحقق من وجود بيانات الصورة الخام (Binary)
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                break
            elif hasattr(part, 'data') and part.data:
                image_bytes = part.data
                break

        if image_bytes:
            print("--- SUCCESS: Image captured! ---")
            # تحويلها لـ Base64 مع ترويسة العرض المباشر للموقع
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded_image}"
        else:
            # استخراج النص الذي أرجعه الموديل بدلاً من الصورة لفهم السبب 
            reason = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content.parts[0], 'text') else "Unknown rejection."
            raise Exception(f"Model returned OK but refused to generate image. Reason: {reason[:100]}")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
