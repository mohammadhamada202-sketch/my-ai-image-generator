import runpod
import os
import base64
import io
from google import genai
from translator_helper import translate_and_optimize

def handler(job):
    try:
        # 1. جلب المفتاح المفعّل (N5UI)
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 2. الترجمة (OpenAI)
        final_optimized_prompt = translate_and_optimize(user_text)
        print(f"Executing with Prompt: {final_optimized_prompt}")

        # 3. طلب التوليد من موديل 2.5 Flash المستقر
        # نستخدم الإعدادات السينمائية التي طلبتها في الـ Prompt
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"{final_optimized_prompt}"
        )

        # 4. المستخرج الذكي (Smart Extractor) لمنع خطأ NoneType
        if not response or not response.candidates:
            raise Exception("No candidates returned from Gemini.")

        image_bytes = None
        # في إصدارات 2026، قد يكون الرد عبارة عن قائمة من الأجزاء (Parts)
        # نبحث في كل الأجزاء عن بيانات الصورة
        for part in response.candidates[0].content.parts:
            # التحقق من inline_data
            if hasattr(part, 'inline_data') and part.inline_data:
                image_bytes = part.inline_data.data
                break
            # التحقق من data مباشرة
            elif hasattr(part, 'data') and part.data:
                image_bytes = part.data
                break
            # التحقق من وجود blob
            elif hasattr(part, 'blob') and part.blob:
                image_bytes = part.blob.data
                break

        if image_bytes:
            print("--- SUCCESS: Image generated and captured! ---")
            return base64.b64encode(image_bytes).decode("utf-8")
        else:
            # إذا أرجع الموديل نصاً (مثلاً رفض بسبب سياسات المحتوى)
            res_text = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content.parts[0], 'text') else "No image found"
            raise Exception(f"Model returned OK but no image data. Reason: {res_text}")

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
