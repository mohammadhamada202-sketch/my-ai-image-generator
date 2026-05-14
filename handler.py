import os
import requests
import base64
import time
import uuid
import runpod
from supabase import create_client

# جلب الإعدادات من RunPod (التي أضفتها أنت)
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN AI - STABILITY ENGINE V1.0 ---")
        
        job_input = job.get('input', {})
        # رابط صورتك الأصلية التي سيتم تحويلها
        init_image_url = job_input.get('image_url') 
        # الوصف (مثلاً: anime style, cyberpunk, pixel art)
        user_prompt = job_input.get('prompt', 'Professional anime style, highly detailed face, masterpiece')

        if not init_image_url:
            return {"error": "Please provide an image_url to transform."}

        # 1. تحميل الصورة الأصلية من الرابط
        print(f"--- [STEP 1] Downloading image: {init_image_url} ---")
        image_response = requests.get(init_image_url)
        if image_response.status_code != 200:
            return {"error": "Failed to download the initial image."}

        # 2. إرسال الطلب لمحرك Stability AI
        print("--- [STEP 2] Transforming to Anime style... ---")
        # نستخدم موديل SDXL 1.0 لقدرته العالية على الأفاتار
        api_host = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
        
        response = requests.post(
            api_host,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            files={
                "init_image": image_response.content
            },
            data={
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 0.40, # 0.40 يحافظ على ملامحك ويضيف لمسة الأنمي
                "text_prompts[0][text]": user_prompt,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30,
            }
        )

        if response.status_code != 200:
            print(f"--- [FAILED] Stability Error: {response.text} ---")
            return {"error": f"Stability API Error: {response.text}"}

        # 3. معالجة بيانات الصورة (Base64 to Bytes)
        data = response.json()
        image_bytes = base64.b64decode(data["artifacts"][0]["base64"])
        print(f"--- [SUCCESS] Transformation complete: {len(image_bytes)/1024:.2f} KB ---")

        # 4. الرفع لـ Supabase
        print("--- [STEP 3] Uploading Avatar to Supabase... ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"avatar_{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        image_url = storage.get_public_url(file_name)
        print(f"--- [DONE] URL: {image_url} ---")

        return {"avatar_url": image_url, "status": "success"}

    except Exception as e:
        print(f"--- [FATAL ERROR] {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
