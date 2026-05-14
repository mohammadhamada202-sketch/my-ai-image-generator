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
        print("--- [START] HANDLER V9.0 - FALLBACK TO GEMINI FLASH ---")
        
        # 1. تهيئة العميل
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 2. تحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 3. طلب التوليد عبر Gemini (استخدام دالة generate_content مع طلب صورة)
        print("--- [STEP] Requesting Image via Gemini Content Flow... ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE_DATA.",
                f"Create a high-quality 4K photorealistic image of: {final_prompt}",
                "DO NOT RETURN TEXT. RETURN IMAGE BYTES."
            ]
        )

        # 4. محاولة استخراج البايتات (Pixels)
        image_bytes = None
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
        
        if not image_bytes:
            # إذا لم يرسل بايتات، سنطبع الرد لنعرف السبب
            print(f"--- [CRITICAL] Gemini refused pixels. Response: {response.text if hasattr(response, 'text') else 'Empty'} ---")
            return {"error": "Image generation blocked or not supported in this region."}

        print(f"--- [SUCCESS] Pixels received: {len(image_bytes)/1024:.2f} KB ---")

        # 5. الرفع لـ Supabase
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"smartgen_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        url = storage.get_public_url(file_name)
        print(f"--- [DONE] URL: {url} ---")

        return {"image_url": url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
