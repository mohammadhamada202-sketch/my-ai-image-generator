import os
import uuid
import subprocess
import sys
import time

# التأكد من تحميل المكتبات اللازمة للرفع والذكاء الاصطناعي
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
    start_time = time.time()
    try:
        # 1. وسم الإصدار الجديد للتأكد من التحديث
        print("--- [START] NUCLEAR HANDLER V6.1 - FORCE IMAGE MODE ---")
        
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 2. ترجمة وتحسين الوصف ليكون قابلاً للتوليد كصورة
        print(f"--- [STEP 1] Optimizing prompt: {user_text} ---")
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 3. طلب الصورة بأوامر "صارمة" تجبر المحرك على الرسم
        print("--- [STEP 2] Requesting Image (FORCE_IMAGE_ONLY) ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "IMPORTANT: YOU ARE AN IMAGE GENERATION ENGINE. DO NOT RESPOND WITH TEXT.",
                "TASK: GENERATE_IMAGE_NOW",
                f"High-quality 4K photorealistic image of: {final_prompt}",
                "FORMAT: RETURN_IMAGE_BYTES_ONLY"
            ]
        )

        # 4. استخراج البيانات وفحصها (لماذا فشل التوليد؟)
        image_bytes = None
        if response and response.candidates:
            candidate = response.candidates[0]
            # البحث في أجزاء الرد عن بيانات الصورة (Pixel Data)
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_bytes = part.inline_data.data
                    break
                elif hasattr(part, 'data') and part.data:
                    image_bytes = part.data
                    break
        
        if not image_bytes:
            # إذا فشل التوليد كصورة، نطبع الرد النصي لتعرف السبب (رفض أمني أو سوء فهم)
            gemini_text = response.text if hasattr(response, 'text') else "No Text Returned"
            print(f"--- [FAILED] Gemini returned text instead of bytes. Message: {gemini_text} ---")
            return {"error": "Gemini sent text instead of an image. Check Logs for reason."}
        
        print(f"--- [SUCCESS] Image received. Size: {len(image_bytes)/1024:.2f} KB ---")

        # 5. الرفع المباشر لـ Supabase (Direct Upload)
        print(f"--- [STEP 3] Uploading to Supabase bucket: {BUCKET_NAME} ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"nuclear_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        image_url = storage.get_public_url(file_name)
        print(f"--- [DONE] Total Process Time: {time.time() - start_time:.2f}s | URL: {image_url} ---")

        return {"image_url": image_url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
