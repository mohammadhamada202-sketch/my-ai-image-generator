import os
import io
import requests
import base64
import time
import runpod
from google import genai
from supabase import create_client

AVATAR_STYLES = {
    "photorealistic": "ultra-detailed professional cinematic portrait",
    "anime": "high-quality anime style, studio ghibli aesthetic",
    "3d_render": "Disney Pixar style 3D avatar"
}

def handler(job):
    try:
        print("--- [START] SMARTGEN - AVATAR TRANSFORMER V1.0 ---")
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        style_key = job_input.get('style', 'photorealistic')

        if not image_url:
            return {"error": "image_url is missing"}

        # 1. تجهيز الصورة
        img_data = requests.get(image_url).content
        img_b64 = base64.b64encode(img_data).decode('utf-8')

        # 2. إعداد Gemini
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        instruction = f"Transform the person in this image into {style_prompt}. Preserve facial identity."
        
        # 3. طلب التحويل
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[instruction, {"mime_type": "image/png", "data": img_b64}]
        )

        # استخراج البكسلات (قد تواجه حظراً في ألمانيا لصور الوجوه)
        try:
            img_raw = response.candidates[0].content.parts[0].inline_data.data
        except:
            return {"error": "Gemini pixels blocked in this region."}

        # 4. الرفع لـ Supabase
        sb = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        file_name = f"avatar_{int(time.time())}.png"
        sb.storage.from_("MyFirstImagesTest1").upload(file_name, img_raw)
        
        return {"avatar_url": sb.storage.from_("MyFirstImagesTest1").get_public_url(file_name)}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
