# handler.py
import os
import base64
import time
import runpod
from supabase import create_client

# استدعاء المكتبة الرسمية لـ Together
from together import Together

# الاستيراد النظامي والمستقر للملفات المساعدة من مستودع الـ GitHub الخاص بك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
from dimensions_config import get_image_dimensions

# جلب متغيرات البيئة من إعدادات RunPod
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN ALL-TOGETHER UNIFIED ENGINE ---")
        job_input = job.get('input', {})
        
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            print("--- [ERROR] Input prompt is missing ---")
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن الذكي (GPT-4o) لتهيئة النص سينمائياً
        print("--- [STEP 1] Activating OpenAI Prompt Engineer (GPT-4o)... ---")
        optimized_prompt = get_epic_prompt(raw_prompt)

        # 2. جلب إعدادات النمط والموديل المخصص له ديناميكياً من ملف الستايلات الخاص بك
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]
        
        final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # 3. استدعاء دالتك الاحترافية لجلب الأبعاد والمقاسات الديناميكية المستقرة
        width, height = get_image_dimensions(job_input)
        print(f"--- [STEP 2] Routed -> Style: {style_key} | Model: {target_model} | Resolution: {width}x{height} ---")

        # 4. التوليد المستقر والسريع عبر مكتبة Together Official SDK
        print(f"--- [STEP 3] Generating via Together Official SDK... ---")
        client = Together()
        
        # تحديد عدد الخطوات تلقائياً بناءً على الموديل (FLUX يحتاج 4 خطوات، بينما SDXL يفضل 20-30 خطوة لتفاصيل الكرتون)
        steps_count = 4 if "flux" in target_model.lower() else 25

        response = client.images.generate(
            model=target_model,
            prompt=final_positive_prompt,
            width=width,
            height=height,
            steps=steps_count,
            response_format="b64_json"
        )
        
        # استخراج البكسلات وفك تشفيرها
        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        # 5. الرفع المستقر والسريع إلى باكت Supabase الخاص بك
        print(f"--- [STEP 4] Uploading Output to Supabase bucket: {BUCKET_NAME} ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم فريد للملف يعتمد على الستايل والوقت الحالي لمنع التكرار
        file_name = f"smartgen_{style_key}_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        # جلب الرابط العام والمباشر لإرساله للموقع
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Image processing complete. Live URL: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "dimensions": f"{width}x{height}",
                "model_used": target_model,
                "steps": steps_count
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المستمر على RunPod والاستماع لطلبات موقعك
runpod.serverless.start({"handler": handler})
