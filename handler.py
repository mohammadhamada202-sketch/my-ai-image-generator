import os
import uuid
import time
import runpod
from google import genai
from supabase import create_client
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] HANDLER V9.0 - FALLBACK MODE ---")
        
        # استخدام العميل العادي
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 1. تحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 2. محاولة التوليد عبر Gemini Flash مع طلب إرجاع "Media"
        print("--- [STEP] Requesting Image via Gemini Media Flow... ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "IMPORTANT: TASK IS IMAGE_GENERATION.",
                f"Generate a high-quality 4K photorealistic image of: {final_prompt}",
                "RETURN_ONLY_IMAGE_DATA"
            ]
        )

        # 3. استخراج البيانات (معالجة احتمالية الرد النصي)
        image_bytes = None
        if response and response.candidates:
            parts = response.candidates[0].content.parts
            for part in parts:
                # التحقق من وجود بيانات ثنائية (Binary Data)
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break
        
        # إذا فشل في إعطاء Bytes، فهذا يعني أن الموديل محظور من الرسم في هذه البيئة
        if not image_bytes:
            err_text = response.text if hasattr(response, 'text') else "Unknown"
            print(f"--- [CRITICAL] Gemini refused to draw. Response: {err_text} ---")
            return {"error": "Model refused to generate pixels in this region."}

        # 4. الرفع لـ Supabase
        print("--- [STEP] Uploading to Supabase... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"fallback_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] URL: {url} ---")

        return {"image_url": url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
