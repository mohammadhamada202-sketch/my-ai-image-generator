import os
import uuid
import subprocess
import sys
import time

# تأكيد استيراد المكتبات
try:
    from supabase import create_client
    print("--- [DIAGNOSTIC] Supabase library loaded ---")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
    from supabase import create_client

import runpod
from google import genai
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    start_time = time.time()
    try:
        # 1. وسم الإصدار (هذا أهم سطر الآن)
        print("--- [START] HANDLER VERSION 5.0 - DIAGNOSTIC MODE ACTIVE ---")
        
        # فحص وجود المفاتيح
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("--- [CRITICAL] SUPABASE_URL or KEY is EMPTY in environment variables! ---")
            return {"error": "Missing Supabase Credentials"}

        # 2. فحص الاتصال بـ Gemini
        print("--- [STEP 1] Initializing Gemini Client... ---")
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        print(f"--- [STEP 2] Optimizing prompt: {user_text} ---")
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Optimized Prompt: {final_prompt} ---")

        # 3. توليد الصورة وفحص حجم البيانات
        print("--- [STEP 3] Requesting Image from Gemini 2.5 Flash... ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[f"Professional high-quality photo: {final_prompt}"]
        )

        image_bytes = None
        if response and response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
        
        if not image_bytes:
            print("--- [FAILED] Gemini returned success but NO IMAGE BYTES found ---")
            return {"error": "Empty image data from Gemini"}
        
        data_size = len(image_bytes) / 1024
        print(f"--- [SUCCESS] Image received. Size: {data_size:.2f} KB ---")

        # 4. محاولة الاتصال والرفع لـ Supabase
        print(f"--- [STEP 4] Connecting to Supabase at: {SUPABASE_URL} ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        file_name = f"diag_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        print(f"--- [STEP 5] Attempting UPLOAD to bucket: {BUCKET_NAME} as {file_name} ---")
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        
        # محاولة الرفع مع صيد الخطأ بدقة
        try:
            upload_response = storage.upload(
                path=file_name,
                file=image_bytes,
                file_options={"content-type": "image/png"}
            )
            print(f"--- [STEP 6] Upload process finished. Response: {upload_response} ---")
        except Exception as upload_err:
            print(f"--- [FAILED] Supabase UPLOAD error: {str(upload_err)} ---")
            raise upload_err

        # 5. استخراج الرابط
        image_url = storage.get_public_url(file_name)
        end_time = time.time()
        print(f"--- [DONE] Total Time: {end_time - start_time:.2f}s | URL: {image_url} ---")

        return {"image_url": image_url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
