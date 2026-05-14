import subprocess
import sys
import os
import uuid
import io  # نحتاجه للتعامل مع البيانات في الذاكرة
from PIL import Image # مكتبة معالجة الصور الموجودة في الـ Dockerfile

# --- [الخطوة الاحتياطية] تثبيت مكتبة Supabase تلقائياً ---
try:
    from supabase import create_client
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
    from supabase import create_client

import runpod
from google import genai
from translator_helper import translate_and_optimize

# إعدادات البيئة
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "MyFirstImagesTest1"

# تهيئة العميل
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def handler(job):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        job_input = job['input']
        user_text = job_input.get('prompt', '')

        # 1. تحسين البرومبت
        final_prompt = translate_and_optimize(user_text)

        # 2. توليد الصورة عبر Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "TASK: GENERATE_IMAGE. NO TEXT OUTPUT. RETURN ONLY THE IMAGE DATA.",
                f"Professional high-quality 4K photo: {final_prompt}"
            ]
        )

        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content.parts:
                image_bytes = None
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        break
                    elif hasattr(part, 'data') and part.data:
                        image_bytes = part.data
                        break
                
                if image_bytes:
                    print("--- [STATUS] Converting image to JPG... ---")
                    
                    # --- تحويل الصورة إلى JPG باستخدام Pillow ---
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # تحويل النمط إلى RGB (ضروري لأن JPG لا يدعم الشفافية)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # حفظ الصورة في "أنبوب" ذاكرة بصيغة JPG
                    output_buffer = io.BytesIO()
                    img.save(output_buffer, format="JPEG", quality=90)
                    jpg_bytes = output_buffer.getvalue()
                    
                    # 3. الرفع لـ Supabase بصيغة JPG
                    file_name = f"smartgen_{uuid.uuid4()}.jpg" # تغيير الامتداد لـ jpg
                    
                    storage = supabase.storage.from_(BUCKET_NAME)
                    storage.upload(
                        path=file_name,
                        file=jpg_bytes,
                        file_options={"content-type": "image/jpeg"}
                    )
                    
                    image_url = storage.get_public_url(file_name)
                    print(f"--- [SUCCESS] JPG Uploaded: {image_url} ---")
                    
                    return {"image_url": image_url}
        
        return {"error": "Image generation or conversion failed."}

    except Exception as e:
        print(f"--- [CRITICAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
