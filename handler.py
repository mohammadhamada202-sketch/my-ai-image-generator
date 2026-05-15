import os
import requests
import base64
import time
import runpod
from supabase import create_client

# الإعدادات من RunPod
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

def handler(job):
    try:
        print("--- [START] TEXT-TO-IMAGE GENERATOR ---")
        job_input = job.get('input', {})
        prompt = job_input.get('prompt', 'A beautiful landscape')

        # طلب الصورة من Stability AI
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {STABILITY_API_KEY}"},
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7, "height": 1024, "width": 1024, "steps": 30,
            }
        )

        if response.status_code != 200: return {"error": response.text}

        # رفع النتيجة لـ Supabase
        img_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"generated_{int(time.time())}.png"
        sb.storage.from_("MyFirstImagesTest1").upload(file_name, img_bytes)
        
        url = sb.storage.from_("MyFirstImagesTest1").get_public_url(file_name)
        return {"image_url": url, "status": "success"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
