# handler.py
import os
import requests
import base64
import time
import runpod
from supabase import create_client

# الاستيراد الدقيق والنظامي بناءً على أسماء ملفاتك ودوالك في مستودع GitHub
from translator_helper import get_epic_prompt
from styles_config import STYLE_ENHANCERS, AVATAR_NEGATIVE_PROMPT
from dimensions_config import get_image_dimensions  # دالتك الاحترافية الجديدة

# جلب إعدادات البيئة من RunPod بعد التحديث
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN NEXT-GEN ENGINE (TOGETHER FLUX) ---")
        job_input = job.get('input', {})
        
        # استلام المدخلات الأساسية من المستخدم والموقع
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')  # النمط الافتراضي

        if not raw_prompt:
            print("--- [ERROR] Input prompt is missing ---")
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن باستخدام دالتك الخاصة (GPT-4o)
        print("--- [STEP 1] Activating OpenAI Prompt Engineer (GPT-4o)... ---")
        optimized_prompt = get_epic_prompt(raw_prompt)

        # 2. استدعاء النمط المناسب من ملف الأنماط الخاص بك (styles_config.py)
        style_details = STYLE_ENHANCERS.get(style_key, STYLE_ENHANCERS["photorealistic"])
        
        # دمج الوصف المطور مع النمط والبرومبت السلبي لضمان الدقة
        final_positive_prompt = f"{optimized_prompt}, {style_details}"

        # 3. استدعاء الأبعاد والمقاسات باستخدام دالتك الاحترافية (dimensions_config.py)
        # نمرر لها الـ job_input بالكامل لتستخرج الـ aspect_ratio تلقائياً
        width, height = get_image_dimensions(job_input)
        
        print(f"--- [STEP 2] Configured -> Style: {style_key} | Resolution: {width}x{height} ---")

        # 4. إرسال الطلب النهائي إلى Together AI لتشغيل موديل FLUX الخارق
        print("--- [STEP 3] Generating pixels via FLUX Engine on Together AI... ---")
        response = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "black-forest-labs/FLUX.1-schnell",  # الموديل الأسرع والأحدث لـ Flux
                "prompt": final_positive_prompt,
                "width": width,
                "height": height,
                "steps": 4,  # موديل Schnell يحتاج 4 خطوات فقط ليعطي تفاصيل خارقة!
                "response_format": "b64_json"
            }
        )

        if response.status_code != 200:
            print(f"--- [FAILED] Together API Error: {response.text} ---")
            return {"error": f"Together API Error: {response.text}"}

        # 5. استخراج البكسلات والرفع إلى Supabase
        print(f"--- [STEP 4] Extracting and Uploading to Supabase bucket: {BUCKET_NAME} ---")
        b64_data = response.json()["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)
        
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم ملف فريد يعتمد على النمط المختار والوقت الحالي لمنع التكرار
        file_name = f"flux_{style_key}_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        # جلب الرابط العام المباشر للصورة الناتجة
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] FLUX Image is live at: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "dimensions": f"{width}x{height}",
                "engine": "FLUX.1 [schnell]"
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المستمر على RunPod
runpod.serverless.start({"handler": handler})
