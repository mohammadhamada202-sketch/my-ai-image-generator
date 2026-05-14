import os
import uuid
import time
import runpod
from google import genai
from supabase import create_client
from translator_helper import translate_and_optimize

# --- إعدادات الاتصال ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] HANDLER V10.0 - NANO BANANA MODE (2.5 FLASH) ---")
        
        # 1. تهيئة العميل باستخدام المسار المستقر
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 2. تحسين الوصف للترجمة
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 3. طلب التوليد (استخدام generate_images المخصصة للصور)
        # ملاحظة: Gemini 2.5 Flash يدعم توليد الصور إذا تم استدعاء الدالة الصحيحة
        print("--- [STEP] Requesting Image from Nano Banana... ---")
        
        # استخدمنا generate_images لأن السجلات السابقة أكدت وجودها في مكتبتك
        response = client.models.generate_images(
            model='gemini-2.5-flash', 
            prompt=f"A high-quality 4K photorealistic image of: {final_prompt}"
        )

        # 4. استخراج البيانات (Bytes)
        image_bytes = None
        if response and hasattr(response, 'generated_images') and response.generated_images:
            image_bytes = response.generated_images[0].image_bytes
            print(f"--- [SUCCESS] Pixels received: {len(image_bytes)/1024:.2f} KB ---")
        else:
            # إذا أعاد نصاً مرة أخرى، سنحاول صيده هنا
            print(f"--- [FAILED] No bytes. Response: {getattr(response, 'text', 'Empty')} ---")
            return {"error": "Nano Banana returned text instead of image data."}

        # 5. الرفع لـ Supabase
        print("--- [STEP] Uploading to Supabase... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"banana_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        url = storage.get_public_url(file_name)
        print(f"--- [DONE] URL: {url} ---")

        return {"image_url": url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
