import os
import uuid
import subprocess
import sys
import time

# 1. تحديث إجباري للمكتبات لأحدث إصدار يدعم Imagen 3
def install_dependencies():
    print("--- [SYSTEM] Force updating libraries... ---")
    try:
        # نقوم بتحديث المكتبة لأحدث إصدار --upgrade
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-genai", "supabase"])
        print("--- [SYSTEM] Libraries updated successfully ---")
    except Exception as e:
        print(f"--- [WARNING] Update failed but continuing: {e} ---")

# تنفيذ التحديث عند تشغيل الملف لأول مرة
install_dependencies()

from google import genai
from supabase import create_client
import runpod
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        # وسم الإصدار الجديد
        print("--- [START] HANDLER V7.2 - FIXING ATTRIBUTE ERROR ---")
        
        # تهيئة العميل مع تحديد الإصدار v1
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # تحسين الوصف
        print(f"--- [STEP 1] Optimizing prompt for: {user_text} ---")
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 2. طلب الصورة (محمي بـ Diagnostic لغرض الفحص)
        print("--- [STEP 2] Requesting Image from Imagen 3... ---")
        
        # فحص وجود الدالة قبل استدعائها لتجنب الانهيار الصامت
        if not hasattr(client.models, 'generate_image'):
            available = [m for m in dir(client.models) if not m.startswith('_')]
            print(f"--- [CRITICAL] Library still outdated! Available methods: {available} ---")
            return {"error": f"Outdated library. Methods found: {available}"}

        response = client.models.generate_image(
            model='imagen-3.0-generate-001',
            prompt=final_prompt
        )

        # 3. استخراج البيانات
        if response and response.generated_images:
            image_bytes = response.generated_images[0].image_bytes
            print(f"--- [SUCCESS] Pixels received: {len(image_bytes)/1024:.2f} KB ---")
        else:
            print("--- [FAILED] Imagen returned no data ---")
            return {"error": "No image data returned"}

        # 4. الرفع لـ Supabase
        print("--- [STEP 3] Uploading to Supabase... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"final_v7_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        image_url = storage.get_public_url(file_name)
        print(f"--- [DONE] URL: {image_url} ---")

        return {"image_url": image_url, "status": "success"}

    except Exception as e:
        error_msg = f"Fatal Error: {str(e)}"
        print(f"--- [ERROR] {error_msg} ---")
        return {"error": error_msg}

# إطلاق السيرفر
runpod.serverless.start({"handler": handler})
