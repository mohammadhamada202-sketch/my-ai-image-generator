import os
import uuid
import subprocess
import sys
import time

# التأكد من المكتبات
try:
    from supabase import create_client
    from google import genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase google-genai"])
    from supabase import create_client
    from google import genai

import runpod
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] HANDLER V7.0 - SWITCHING TO IMAGEN (REAL ARTIST) ---")
        
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 1. تحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Prompt: {final_prompt} ---")

        # 2. توليد الصورة باستخدام Imagen (وليس Gemini Flash)
        print("--- [STEP] Requesting Image from Imagen 3... ---")
        
        # ملاحظة: نستخدم generate_image هنا
        response = client.models.generate_image(
            model='imagen-3.0-generate-001', # هذا هو الموديل الذي يرسم فعلياً
            prompt=final_prompt
        )

        # 3. استخراج بيانات الصورة
        image_bytes = None
        if response and response.generated_images:
            # Imagen يعيد الصورة في قائمة generated_images
            image_bytes = response.generated_images[0].image_bytes
        
        if not image_bytes:
            print("--- [FAILED] Imagen did not return any bytes ---")
            return {"error": "Imagen failed to generate image."}
        
        print(f"--- [SUCCESS] Image generated! Size: {len(image_bytes)/1024:.2f} KB ---")

        # 4. الرفع لـ Supabase
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"imagen_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        image_url = storage.get_public_url(file_name)
        print(f"--- [DONE] URL: {image_url} ---")

        return {"image_url": image_url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
