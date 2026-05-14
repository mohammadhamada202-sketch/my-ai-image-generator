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
        print("--- [START] HANDLER V8.0 - FINAL SUCCESS VERSION ---")
        
        # تهيئة العميل
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
        
        job_input = job.get('input', {})
        user_text = job_input.get('prompt', 'Apple')
        
        # 1. ترجمة وتحسين الوصف
        final_prompt = translate_and_optimize(user_text)
        print(f"--- [DEBUG] Final Prompt: {final_prompt} ---")

        # 2. طلب الصورة من Imagen 3 (باستخدام الدالة الصحيحة المكتشفة s)
        print("--- [STEP] Requesting Image from Imagen 3 Artist... ---")
        
        # هنا استخدمنا 'generate_images' كما ظهرت في الـ Logs الخاصة بك
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=final_prompt
        )

        # 3. استخراج البكسلات
        image_bytes = None
        if response and hasattr(response, 'generated_images') and response.generated_images:
            image_bytes = response.generated_images[0].image_bytes
            print(f"--- [SUCCESS] Real Image Received: {len(image_bytes)/1024:.2f} KB ---")
        else:
            print("--- [FAILED] No image bytes in response ---")
            return {"error": "Imagen returned success but no data"}

        # 4. الرفع المباشر لـ Supabase
        print("--- [STEP] Uploading to Supabase... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"success_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})

        url = storage.get_public_url(file_name)
        print(f"--- [DONE] SUCCESS! URL: {url} ---")

        return {"image_url": url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
