import os
import requests
import base64
import time
import runpod
from supabase import create_client

# جلب الإعدادات مع تنظيف فائق للرابط
def get_clean_env(key):
    val = os.environ.get(key, "").strip()
    # إزالة أي علامات اقتباس قد تكون دخلت بالخطأ
    return val.replace('"', '').replace("'", "")

STABILITY_API_KEY = get_clean_env("STABILITY_API_KEY")
# التأكد من أن الرابط يبدأ بـ https:// ولا ينتهي بـ /
SUPABASE_URL = get_clean_env("SUPABASE_URL").rstrip('/')
SUPABASE_KEY = get_clean_env("SUPABASE_KEY")
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print(f"--- [DEBUG] Connecting to: {SUPABASE_URL} ---")
        
        job_input = job.get('input', {})
        prompt = job_input.get('prompt')

        if not prompt:
            return {"error": "Prompt is missing"}

        # 1. طلب التوليد من Stability AI
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {STABILITY_API_KEY}"},
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7, "height": 1024, "width": 1024, "steps": 30,
            }
        )

        if response.status_code != 200:
            return {"error": f"Stability API Error: {response.text}"}

        # 2. معالجة البكسلات
        image_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        print(f"--- [SUCCESS] Image Generated ({len(image_bytes)/1024:.2f} KB) ---")

        # 3. الرفع لـ Supabase مع معالجة أخطاء الاتصال
        print(f"--- [STEP] Uploading to bucket: {BUCKET_NAME} ---")
        try:
            # تهيئة العميل
            sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            file_name = f"final_{int(time.time())}.png"
            
            storage = sb_client.storage.from_(BUCKET_NAME)
            storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
            
            public_url = storage.get_public_url(file_name)
            print(f"--- [DONE] URL: {public_url} ---")
            return {"image_url": public_url, "status": "success"}
            
        except Exception as conn_error:
            # هنا سيظهر لك تفاصيل أكثر عن سبب فشل الاتصال بسوبابيس
            print(f"--- [CONNECTION ERROR] Check your SUPABASE_URL: {str(conn_error)} ---")
            return {"error": f"Supabase Connection Failed: {str(conn_error)}"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
