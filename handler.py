import os
import requests
import base64
import time
import runpod
from supabase import create_client

# دالة لتنظيف متغيرات البيئة من المسافات أو علامات الاقتباس الزائدة
def get_clean_env(key):
    val = os.environ.get(key, "").strip()
    return val.replace('"', '').replace("'", "")

# جلب الإعدادات المنظفة
STABILITY_API_KEY = get_clean_env("STABILITY_API_KEY")
SUPABASE_URL = get_clean_env("SUPABASE_URL").rstrip('/') # إزالة السلاش في النهاية
SUPABASE_KEY = get_clean_env("SUPABASE_KEY")
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        # طباعة الرابط للتأكد منه في السجلات (Debug)
        print(f"--- [DEBUG] Target Supabase URL: {SUPABASE_URL} ---")
        
        job_input = job.get('input', {})
        prompt = job_input.get('prompt', 'red Apple')

        # 1. طلب التوليد من Stability AI
        print(f"--- [STEP 1] Generating image for: {prompt} ---")
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7, "height": 1024, "width": 1024, "steps": 30,
            }
        )

        if response.status_code != 200:
            return {"error": f"Stability Error: {response.text}"}

        # 2. معالجة البكسلات
        image_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        print(f"--- [SUCCESS] Pixels received: {len(image_bytes)/1024:.2f} KB ---")

        # 3. الرفع لـ Supabase
        print(f"--- [STEP 2] Attempting upload to: {BUCKET_NAME} ---")
        try:
            # تهيئة العميل بالرابط المنظف
            sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            file_name = f"gen_{int(time.time())}.png"
            
            storage = sb_client.storage.from_(BUCKET_NAME)
            storage.upload(
                path=file_name, 
                file=image_bytes, 
                file_options={"content-type": "image/png"}
            )
            
            public_url = storage.get_public_url(file_name)
            print(f"--- [DONE] Public URL: {public_url} ---")
            return {"image_url": public_url, "status": "success"}

        except Exception as sb_err:
            print(f"--- [SUPABASE CONNECTION ERROR] ---: {str(sb_err)}")
            return {"error": f"Connection Failed: {str(sb_err)}"}

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
