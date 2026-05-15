import os
import requests
import base64
import time
import runpod
from supabase import create_client

# الإعدادات
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

def handler(job):
    try:
        print("--- [START] SMARTGEN - IMAGE GENERATOR V1.0 ---")
        job_input = job.get('input', {})
        prompt = job_input.get('prompt')

        if not prompt:
            return {"error": "Prompt is missing in input"}

        # طلب التوليد من Stability AI
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {STABILITY_API_KEY}"},
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7, "height": 1024, "width": 1024, "steps": 30,
            }
        )

        if response.status_code != 200:
            return {"error": response.text}

        # الرفع لـ Supabase
        img_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"gen_{int(time.time())}.png"
        
        storage = sb.storage.from_("MyFirstImagesTest1")
        storage.upload(path=file_name, file=img_bytes, file_options={"content-type": "image/png"})
        
        return {"image_url": storage.get_public_url(file_name), "status": "success"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
