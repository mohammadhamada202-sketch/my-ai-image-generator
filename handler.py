# handler.py
import os
import requests
import base64
import time
import runpod
from supabase import create_client

# الاستيراد الدقيق والنظامي بناءً على أسماء ملفاتك ودوالك في مستودع GitHub
from translator_helper import translate_and_optimize
from styles_config import STYLE_ENHANCERS, AVATAR_NEGATIVE_PROMPT
from dimensions_config import get_image_dimensions  # دالتك الاحترافية الجديدة

# جلب إعدادات البيئة من RunPod
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN PRODUCTION ENGINE V19.0 ---")
        job_input = job.get('input', {})
        
        # استلام المدخلات الأساسية من المستخدم
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')  # النمط الافتراضي: photorealistic

        if not raw_prompt:
            print("--- [ERROR] Input prompt is missing ---")
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن (الذي يعتمد على GPT-4o الخرافي في ملفك)
        print("--- [STEP 1] Activating OpenAI Prompt Engineer... ---")
        optimized_prompt = translate_and_optimize(raw_prompt)

        # 2. استدعاء النمط المناسب من ملف الأنماط الخاص بك (styles_config.py)
        style_details = STYLE_ENHANCERS.get(style_key, STYLE_ENHANCERS["photorealistic"])
        final_positive_prompt = f"{optimized_prompt}, {style_details}"

        # 3. استدعاء الأبعاد والمقاسات باستخدام دالتك الاحترافية (dimensions_config.py)
        # نمرر لها الـ job_input بالكامل لتستخرج الـ aspect_ratio تلقائياً
        width, height = get_image_dimensions(job_input)
        
        print(f"--- [STEP 2] Configured -> Style: {style_key} | Resolution: {width}x{height} ---")

        # 4. إرسال الطلب النهائي إلى Stability AI بالتنسيق والمقاس والبرومبت السلبي المعتمد عندك
        print("--- [STEP 3] Generating pixels via Stability AI... ---")
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [
                    {"text": final_positive_prompt, "weight": 1.0},
                    {"text": AVATAR_NEGATIVE_PROMPT, "weight": -1.0}  # البرومبت السلبي الموحد من ملفك
                ],
                "cfg_scale": 8.5,
                "height": height,
                "width": width,
                "steps": 40,  # عدد خطوات عالي لضمان تفاصيل خرافية وجودة فائقة
            }
        )

        if response.status_code != 200:
            print(f"--- [FAILED] Stability API Error: {response.text} ---")
            return {"error": f"Stability Error: {response.text}"}

        # 5. معالجة البكسلات الناتجة والرفع إلى Supabase
        print(f"--- [STEP 4] Uploading to Supabase bucket: {BUCKET_NAME} ---")
        image_bytes = base64.b64decode(response.json()["artifacts"][0]["base64"])
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم ملف فريد يعتمد على النمط المختار والوقت الحالي لمنع التكرار
        file_name = f"smartgen_{style_key}_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        # جلب الرابط العام المباشر للصورة
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Styled image is live at: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "dimensions": f"{width}x{height}"
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المستمر على RunPod
runpod.serverless.start({"handler": handler})
