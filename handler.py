# handler.py
import api_key 

import os
import base64
import time
import runpod
from supabase import create_client

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS, AVATAR_NEGATIVE_PROMPT
from dimensions_config import get_image_dimensions

# إعدادات الروابط والمفاتيح
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "hip-gecko-496121-f9").strip()
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1").strip()

# 🛠️ دالة تحديث الحالة اللحظية في جدول جدار الحماية بـ Supabase
def update_live_status(sb_client, job_id, status_name, error_msg=None):
    try:
        sb_client.table("job_status_tracker").upsert({
            "job_id": job_id,
            "status": status_name,
            "error_message": error_msg,
            "updated_at": "now()"
        }).execute()
        print(f"--- [REALTIME STATUS] -> {status_name} ---")
    except Exception as e:
        print(f"--- [REALTIME ERROR] Failed to update status: {str(e)} ---")

def handler(job):
    # لقط معرف المهمة الفريد المولد من ران بود فوراً
    job_id = job.get('id')
    sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        print(f"--- [START] SMARTGEN ENGINE FOR JOB: {job_id} ---")
        
        # ⚙️ الخطوة 1: الإعداد وتجهيز الحاوية
        update_live_status(sb_client, job_id, "initializing")
        
        job_input = job.get('input', {})
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            update_live_status(sb_client, job_id, "failed", "Prompt is missing.")
            return {"error": "Prompt is missing."}

        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]

        # 🧠 الخطوة 2: الترجمة والتحسين عبر OpenAI Prompt Engineer
        update_live_status(sb_client, job_id, "translating")
        try:
            optimized_prompt = get_epic_prompt(raw_prompt)
            print(f"--- [SUCCESS] Prompt Optimized to: '{optimized_prompt}' ---")
        except Exception as trans_err:
            print(f"--- [WARNING] Translator failed: {str(trans_err)} | Using raw prompt ---")
            optimized_prompt = raw_prompt

        famous_characters = ["tom and jerry", "tom & jerry", "mickey mouse", "spiderman", "superman", "batman", "popeye", "pikachu", "goku"]
        is_famous = any(char in optimized_prompt.lower() for char in famous_characters) or any(char in raw_prompt.lower() for char in famous_characters)

        if style_key == "cartoon" or is_famous:
            print(f"--- [INFO] Safety Bypass applied for cartoon/famous character ---")
            final_positive_prompt = f"stylized fan-art character artistic illustration of {optimized_prompt}, {style_enhancer}"
        else:
            final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # 🎨 الخطوة 3: التوليد داخل سيرفرات جوجل (Imagen 3)
        update_live_status(sb_client, job_id, "generating")
        
        width, height = get_image_dimensions(job_input)
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        
        aspect_ratio = "1:1"
        if width > height: aspect_ratio = "16:9"
        elif height > width: aspect_ratio = "9:16"

        model = ImageGenerationModel.from_pretrained(target_model)
        response = model.generate_images(
            prompt=final_positive_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="block_few"
        )
        
        if not response.images:
            update_live_status(sb_client, job_id, "failed", "Blocked by Google Safety Filters.")
            return {"error": "The prompt was flagged by Google Safety Filters."}
            
        image_bytes = response.images[0]._image_bytes

        # 📦 الخطوة 4: حفظ الصورة ورفعها إلى الباكت
        update_live_status(sb_client, job_id, "uploading")
        
        file_name = f"smartgen_{style_key}_{int(time.time())}.png"
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        public_url = storage.get_public_url(file_name)
        
        # 🎉 الخطوة 5: النجاح النهائي والاكتمال
        update_live_status(sb_client, job_id, "success")
        print(f"--- [SUCCESS] Live URL: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "model_used": target_model,
                "aspect_ratio_used": aspect_ratio,
                "provider": "google"
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        update_live_status(sb_client, job_id, "failed", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
