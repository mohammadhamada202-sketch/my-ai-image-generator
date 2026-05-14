import os
import uuid
import subprocess
import sys
import time

# 1. التأكد من تثبيت النسخة الصحيحة من المكتبة
try:
    from google import genai
    from supabase import create_client
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai supabase"])
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
        print("--- [START] HANDLER V7.1 - FIXING IMAGEN FUNCTION CALL ---")
        
        # تهيئة العميل
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # تحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 2. طلب الصورة (الطريقة البديلة المتوافقة)
        print("--- [STEP] Requesting Image from Imagen 3... ---")
        
        # في بعض النسخ، يتم الوصول للموديل عبر التسمية الكاملة مباشرة
        response = client.models.generate_image(
            model='imagen-3.0-generate-001',
            prompt=final_prompt,
            config=None # يمكنك إضافة إعدادات إضافية هنا لاحقاً
        )

        # 3. استخراج البيانات
        # ملاحظة: إذا استمر الخطأ، سنقوم بتغيير السطر أدناه لاستخدام الدالة الخام
        image_bytes = response.generated_images[0].image_bytes
        
        if not image_bytes:
            raise Exception("No image bytes returned from Imagen")

        print(f"--- [SUCCESS] Image size: {len(image_bytes)/1024:.2f} KB ---")

        # 4. الرفع لـ Supabase
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"final_{int(time.time())}.png"
        
        storage = sb.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        url = storage.get_public_url(file_name)
        print(f"--- [DONE] Image live at: {url} ---")

        return {"image_url": url, "status": "success"}

    except Exception as e:
        # إذا فشل التوليد بسبب الدالة، سنطبع قائمة الدوال المتاحة لنعرف الاسم الصحيح
        available_methods = [method for method in dir(client.models) if not method.startswith('_')]
        error_msg = f"Error: {str(e)} | Available methods: {available_methods}"
        print(f"--- [FATAL] {error_msg} ---")
        return {"error": error_msg}

runpod.serverless.start({"handler": handler})
