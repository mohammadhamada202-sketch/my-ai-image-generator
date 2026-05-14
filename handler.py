import subprocess
import sys
import os
import uuid
import base64

# --- [الخطوة الاحتياطية] تثبيت مكتبة Supabase تلقائياً إذا لم تكن موجودة ---
try:
    from supabase import create_client
except ImportError:
    print("--- [INFO] Supabase library not found. Installing now... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
    from supabase import create_client

import runpod
from google import genai
from translator_helper import translate_and_optimize

# --- [إعدادات البيئة] تأكد من إضافتها في RunPod Dashboard ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # استخدم الـ Secret Key لضمان الصلاحيات
BUCKET_NAME = "MyFirstImagesTest1"

# تهيئة عميل Supabase
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("--- [WARNING] Supabase credentials missing! ---")

def handler(job):
    try:
        # إعداد عميل Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        print("--- [STATUS] Step 1: Optimizing prompt... ---")
        final_prompt = translate_and_optimize(user_text)

        print(f"--- [STATUS] Step 2: Generating image via Gemini 2.5 ---")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality 4K photo: {final_prompt}"
            ]
        )

        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                image_bytes = None
                
                # البحث عن بيانات الصورة في الرد
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        break
                    elif hasattr(part, 'data') and part.data:
                        image_bytes = part.data
                        break
                
                if image_bytes:
                    print("--- [STATUS] Step 3: Uploading to Supabase Storage... ---")
                    
                    # إنشاء اسم ملف فريد (مثلاً: gen_123e4567.png)
                    file_name = f"smartgen_{uuid.uuid4()}.png"
                    
                    # رفع الصورة إلى البكت المخصص
                    storage = supabase.storage.from_(BUCKET_NAME)
                    storage.upload(
                        path=file_name,
                        file=image_bytes,
                        file_options={"content-type": "image/png"}
                    )
                    
                    # الحصول على الرابط المباشر
                    image_url = storage.get_public_url(file_name)
                    
                    print(f"--- [SUCCESS] Image live at: {image_url} ---")
                    
                    # إرجاع الرابط فقط لتجنب تعليق المتصفح في v0
                    return {"image_url": image_url}
                
                else:
                    return {"error": "Model sent text instead of image data."}
        
        return {"error": "No image data found in response."}

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
