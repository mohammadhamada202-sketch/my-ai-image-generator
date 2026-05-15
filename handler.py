import os
import requests
import base64
import time
import runpod
from supabase import create_client

# جلب الإعدادات من Environment Variables في RunPod
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN - IMAGE GENERATOR V1.0 ---")
        
        # 1. استلام المدخلات
        job_input = job.get('input', {})
        prompt = job_input.get('prompt')

        if not prompt:
            print("--- [ERROR] No prompt provided in input ---")
            return {"error": "Prompt is missing. Please provide a text prompt."}

        print(f"--- [STEP 1] Generating image for prompt: {prompt} ---")

        # 2. طلب التوليد من Stability AI (المسار المستقر في ألمانيا)
        # نستخدم موديل SDXL 1.0 لضمان جودة 1024x1024
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            }
        )

        if response.status_code != 200:
            error_detail = response.text
            print(f"--- [FAILED] Stability API Error: {error_detail} ---")
            return {"error": f"Stability API Error: {error_detail}"}

        # 3. معالجة البكسلات (Base64 to Bytes)
        data = response.json()
        image_bytes = base64.b64decode(data["artifacts"][0]["base64"])
        print(f"--- [SUCCESS] Pixels received: {len(image_bytes)/1024:.2f} KB ---")

        # 4. الرفع لـ Supabase
        print(f"--- [STEP 2] Uploading to Supabase bucket: {BUCKET_NAME} ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم فريد للملف لمنع التكرار
        file_name = f"gen_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        # 5. الحصول على الرابط العام
        public_url = storage.get_public_url(file_name)
        print(f"--- [DONE] Image is live at: {public_url} ---")

        return {
            "image_url": public_url,
            "status": "success",
            "file_name": file_name
        }

    except Exception as e:
        error_msg = str(e)
        print(f"--- [FATAL ERROR] {error_msg} ---")
        return {"error": error_msg}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
