import os
import uuid
import subprocess
import sys
import time

# 1. التأكد من تثبيت المكتبات اللازمة داخل الحاوية
try:
    from supabase import create_client
    from google import genai
    print("--- [SYSTEM] Libraries loaded successfully ---")
except ImportError:
    print("--- [SYSTEM] Installing missing libraries... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase google-genai"])
    from supabase import create_client
    from google import genai

import runpod
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال (Environment Variables) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    start_time = time.time()
    try:
        # وسم الإصدار للتأكد من سحب الكود الجديد في الـ Logs
        print("--- [START] HANDLER V7.0 - IMAGEN 3 GENERATION MODE ---")
        
        # تهيئة عميل Google GenAI
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 2. ترجمة وتحسين الوصف
        print(f"--- [STEP 1] Optimizing prompt for: {user_text} ---")
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 3. توليد الصورة (باستخدام Imagen وليس Gemini Flash)
        # هذا هو التعديل الجوهري الذي سيخصم من الرصيد ويرسل بكسلات حقيقية
        print("--- [STEP 2] Calling Imagen 3 Artist... ---")
        response = client.models.generate_image(
            model='imagen-3.0-generate-001',
            prompt=final_prompt
        )

        # 4. استخراج بيانات الصورة (Bytes)
        image_bytes = None
        if response and response.generated_images:
            # Imagen يعيد الصورة مباشرة كـ bytes في أول عنصر من القائمة
            image_bytes = response.generated_images[0].image_bytes
        
        if not image_bytes:
            print("--- [FAILED] Imagen returned success but NO BYTES found ---")
            return {"error": "Failed to generate image bytes."}
        
        print(f"--- [SUCCESS] Real Image Data Received! Size: {len(image_bytes)/1024:.2f} KB ---")

        # 5. الرفع المباشر لـ Supabase
        print(f"--- [STEP 3] Connecting to Supabase for Upload... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم فريد للملف
        file_name = f"smartgen_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        print(f"--- [STATUS] Uploading {file_name} to bucket: {BUCKET_NAME} ---")
        
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        # 6. الحصول على الرابط النهائي
        image_url = storage.get_public_url(file_name)
        print(f"--- [DONE] Image is LIVE at: {image_url} ---")

        return {
            "image_url": image_url, 
            "status": "success",
            "execution_time": f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        error_msg = f"Nuclear Error: {str(e)}"
        print(f"--- [FATAL] {error_msg} ---")
        return {"error": error_msg}

# إطلاق السيرفر
runpod.serverless.start({"handler": handler})
