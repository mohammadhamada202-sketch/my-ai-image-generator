import os
import io
import uuid
import subprocess
import sys
from PIL import Image

# محاولة استيراد مكتبة supabase وتثبيتها تلقائياً
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
    print("--- [ERROR] Supabase credentials missing from environment! ---")
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def handler(job):
    try:
        # تأكيد إصدار الكود في السجلات
        print("--- [SYSTEM] STARTING HANDLER VERSION 3.0 ---")
        
        # 1. تهيئة عميل Gemini
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', 'A beautiful landscape')

        # 2. الترجمة وتحسين الوصف
        print(f"--- [STATUS] Optimizing prompt for: {user_text} ---")
        final_prompt = translate_and_optimize(user_text) [cite: 32]
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 3. توليد الصورة عبر Gemini
        print("--- [STATUS] Calling Gemini to generate image... ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality 4K photo: {final_prompt}"
            ]
        )

        image_bytes = None
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        break
                    elif hasattr(part, 'data') and part.data:
                        image_bytes = part.data
                        break

        if not image_bytes:
            print("--- [ERROR] No image data found in Gemini response ---")
            return {"error": "Failed to receive image data from Gemini."}

        # 4. معالجة الصورة وتحويلها إلى JPG
        print("--- [STATUS] Converting PNG to JPG for compatibility... ---")
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=90)
        jpg_data = output_buffer.getvalue()

        # 5. الرفع إلى Supabase
        file_name = f"smartgen_{uuid.uuid4()}.jpg"
        print(f"--- [STATUS] Uploading {file_name} to Supabase Bucket: {BUCKET_NAME} ---")
        
        storage = supabase.storage.from_(BUCKET_NAME)
        upload_result = storage.upload(
            path=file_name,
            file=jpg_data,
            file_options={"content-type": "image/jpeg"}
        )
        print(f"--- [DEBUG] Upload Result: {upload_result} ---")

        # 6. الحصول على الرابط العام
        image_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Image live at: {image_url} ---")

        return {
            "image_url": image_url,
            "status": "success",
            "prompt_used": final_prompt
        }

    except Exception as e:
        error_msg = f"Critical Error: {str(e)}"
        print(f"--- [ERROR] {error_msg} ---")
        return {"error": error_msg}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
