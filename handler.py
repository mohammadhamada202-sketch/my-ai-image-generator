# handler.py
import os
import base64
import time
import runpod
from supabase import create_client
from together import Together

# الاستيراد المباشر من ملفات مشروعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
from dimensions_config import get_image_dimensions

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN CLEAN TUNED ENGINE ---")
        job_input = job.get('input', {})
        
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن
        optimized_prompt = get_epic_prompt(raw_prompt)

        # 2. جلب إعدادات النمط والموديل ديناميكياً
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]
        
        # ⚠️ حيلة جراحية: لو الستايل كرتون، نقوم بتنظيف النص القادم من المترجم من أي كلمات واقعية أو لمعان
        if style_key == "cartoon":
            # تنظيف النص المترجم لضمان عدم وجود كلمات تدعو للواقعية والتجسيم
            words_to_remove = ["highly detailed", "hyper-detailed", "cinematic", "photorealistic", "ultra realistic", "8k", "masterpiece render", "3d render"]
            cleaned_prompt = optimized_prompt.lower()
            for word in words_to_remove:
                cleaned_prompt = cleaned_prompt.replace(word, "")
            
            # دمج النص النظيف مع محسن الكرتون الصارم جداً
            final_positive_prompt = f"{cleaned_prompt}, {style_enhancer}"
        else:
            final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # 3. المقاسات والأبعاد
        width, height = get_image_dimensions(job_input)
        print(f"--- [ROUTE] Style: {style_key} | Model: {target_model} ---")

        # 4. التوليد عبر Together AI
        client = Together()
        
        # إذا كان الموديل هو SDXL نرفع الخطوات لـ 28 ليعطي تفاصيل رسم مسطح ممتاز
        if "stable-diffusion" in target_model.lower():
            steps_count = 28
            print(f"--- [ENGINE] Launching SDXL 2D Engine with {steps_count} steps... ---")
        else:
            steps_count = 4
            print(f"--- [ENGINE] Launching FLUX Realism Engine with {steps_count} steps... ---")

        response = client.images.generate(
            model=target_model,
            prompt=final_positive_prompt,
            width=width,
            height=height,
            steps=steps_count,
            response_format="b64_json"
        )
        
        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        # 5. الرفع إلى باكت Supabase الخاص بك
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        file_name = f"smartgen_{style_key}_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Processing complete. Live URL: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "model_used": target_model,
                "steps": steps_count
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
