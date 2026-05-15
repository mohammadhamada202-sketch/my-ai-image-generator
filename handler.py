import os
import requests
import base64
import time
import runpod
from openai import OpenAI
from supabase import create_client

# --- إعدادات البيئة ---
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

# --- دالة التحسين والترجمة الخاصة بك ---
def translate_and_optimize(user_input):
    if not user_input or user_input.strip() == "":
        return user_input

    if not OPENAI_API_KEY:
        print("CRITICAL: OPENAI_API_KEY is missing.")
        return user_input

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional prompt engineer. Translate the input to English and enhance it with artistic details for high-quality AI image generation. Return ONLY the final English prompt."
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        optimized_text = response.choices[0].message.content
        print(f"--- [AI ENHANCED] ---: {optimized_text}")
        return optimized_text
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return user_input

# --- الدالة الرئيسية للـ RunPod ---
def handler(job):
    try:
        print("--- [START] SMARTGEN ENGINE V14.0 ---")
        job_input = job.get('input', {})
        raw_user_prompt = job_input.get('prompt', 'red apple')

        # 1. الترجمة والتحسين عبر OpenAI
        print(f"--- [STEP 1] Optimizing Prompt: {raw_user_prompt} ---")
        optimized_prompt = translate_and_optimize(raw_user_prompt)

        # 2. توليد الصورة عبر Stability AI
        print("--- [STEP 2] Generating Image via Stability AI ---")
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [{"text": optimized_prompt}],
                "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1, "steps": 30,
            }
        )

        if response.status_code != 200:
            return {"error": f"Stability Error: {response.text}"}

        # 3. معالجة البيانات والرفع لـ Supabase
        image_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"final_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Public URL: {public_url} ---")
        
        return {
            "image_url": public_url,
            "optimized_prompt": optimized_prompt,
            "status": "success"
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
