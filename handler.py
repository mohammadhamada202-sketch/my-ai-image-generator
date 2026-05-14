import subprocess
import sys
import os
import uuid

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

        # الخطوة 1: تحسين البرومبت
        final_prompt = translate_and_optimize(user_text)

        # الخطوة 2: توليد الصورة
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
                    # الخطوة 3: الرفع المباشر (بدون تحويل لـ Base64)
                    file_name = f"smartgen_{uuid.uuid4()}.png"
                    
                    storage = supabase.storage.from_(BUCKET_NAME)
                    storage.upload(
                        path=file_name,
                        file=image_bytes,
                        file_options={"content-type": "image/png"}
                    )
                    
                    image_url = storage.get_public_url(file_name)
                    
                    return {"image_url": image_url}
        
        return {"error": "Image generation failed."}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
