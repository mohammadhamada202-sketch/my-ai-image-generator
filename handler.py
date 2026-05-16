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
        print("--- [START] SMARTGEN CLEAN CARTOON ENGINE ---")
        job_input = job.get('input', {})
        
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            return {"error": "Prompt is missing."}

        # جلب إعدادات النمط والموديل ديناميكياً
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]

        # 🔥 خطوة حاسمة: لو الستايل كرتون، نلغي المترجم تماماً ونمرر نص المستخدم صافي 
        # لكي نمنع الـ GPT من إضافة أي تفاصيل سينمائية أو أنمي
        if style_key == "cartoon":
            print("--- [INFO] Cartoon detected: Passing raw prompt directly to prevent anime distortion ---")
            final_positive_prompt = f"{raw_prompt}, {style_enhancer}"
        else:
            print("--- [INFO] Standard Style: Activating OpenAI Prompt Engineer ---")
            optimized_prompt = get_epic_prompt(raw_prompt)
            final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # المقاسات والأبعاد
        width, height = get_image_dimensions(job_input)
        print(f"--- [ROUTE] Style: {style_key} | Model: {target_model} | Resolution: {width}x{height} ---")

        # التوليد عبر Together AI
        client = Together()
        
        # تشغيل SDXL بـ 28 خطوة لضمان النعومة والخطوط النظيفة للكرتون
        steps_count = 28 if "stable-diffusion" in target_model.lower() else 4

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

        # الرفع إلى باكت Supabase الخاص بك
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
