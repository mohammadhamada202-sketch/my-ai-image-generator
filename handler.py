import os
import uuid
import subprocess
import sys

# التأكد من وجود مكتبة supabase
try:
    from supabase import create_client
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

# تهيئة عميل Supabase
if not SUPABASE_URL or not SUPABASE_KEY:
    print("--- [ERROR] Supabase credentials missing! ---")
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def handler(job):
    try:
        # رقم إصدار جديد للتأكد من التحديث في الـ Logs
        print("--- [SYSTEM] STARTING HANDLER V4.0 - DIRECT UPLOAD ---")
        
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        job_input = job['input']
        user_text = job_input.get('prompt', 'A beautiful landscape')

        # 1. تحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 2. توليد الصورة
        print("--- [STATUS] Calling Gemini... ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT.",
                f"Professional high-quality 4K photo: {final_prompt}"
            ]
        )

        image_bytes = None
        if response and response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break

        if not image_bytes:
            return {"error": "No image data from Gemini."}

        # 3. الرفع المباشر (بدون تحويل)
        file_name = f"smartgen_{uuid.uuid4()}.png" # Gemini يعيد PNG افتراضياً
        print(f"--- [STATUS] Direct Uploading {file_name} to Supabase... ---")
        
        storage = supabase.storage.from_(BUCKET_NAME)
        # نرسل image_bytes كما هي تماماً
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        image_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Image live at: {image_url} ---")

        return {"image_url": image_url, "status": "success"}

    except Exception as e:
        print(f"--- [ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
