# handler.py
import os
import base64
import time
import runpod
from supabase import create_client

# استدعاء مكتبة Together الرسمية التي تعتمد عليها
from together import Together

# الاستيراد الدقيق والنظامي بناءً على أسماء ملفاتك ودوالك في مستودع GitHub
from translator_helper import get_epic_prompt
from styles_config import STYLE_ENHANCERS
from dimensions_config import get_image_dimensions  # دالتك الاحترافية للمقاسات

# جلب إعدادات بيئة التخزين لـ Supabase من RunPod
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN NEXT-GEN ENGINE (TOGETHER SDK FLUX) ---")
        job_input = job.get('input', {})
        
        # استلام المدخلات الأساسية من المستخدم والموقع
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')  # النمط الافتراضي في حال لم يرسل الموقع نمطاً

        if not raw_prompt:
            print("--- [ERROR] Input prompt is missing ---")
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن باستخدام دالتك الخاصة المتصلة بـ (GPT-4o)
        print("--- [STEP 1] Activating OpenAI Prompt Engineer (GPT-4o)... ---")
        optimized_prompt = get_epic_prompt(raw_prompt)

        # 2. استدعاء النمط المناسب من ملف الأنماط الخاص بك (styles_config.py)
        style_details = STYLE_ENHANCERS.get(style_key, STYLE_ENHANCERS["photorealistic"])
        final_positive_prompt = f"{optimized_prompt}, {style_details}"

        # 3. استدعاء الأبعاد والمقاسات الديناميكية باستخدام دالتك (dimensions_config.py)
        # نمرر الـ job_input كاملاً لتستخرج الدالة الـ aspect_ratio وتحدد الطول والعرض تلقائياً
        width, height = get_image_dimensions(job_input)
        
        print(f"--- [STEP 2] Configured -> Style: {style_key} | Resolution: {width}x{height} ---")

        # 4. التوليد عبر مكتبة Together الرسمية بموديل FLUX الخارق
        print("--- [STEP 3] Generating pixels via Together Official SDK... ---")
        client = Together()  # يتم الربط تلقائياً بـ TOGETHER_API_KEY الموجود في بيئة الـ RunPod
        
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell",  # موديل فلوكس السريع والموفر جداً للميزانية
            prompt=final_positive_prompt,              # البرومبت النهائي المترجم والمضاف إليه النمط الفني
            width=width,                               # العرض المستخرج من دالتك
            height=height,                             # الطول المستخرج من دالتك
            steps=4,                                   # موديل فلوكس يحتاج 4 خطوات فقط ليعطي تفاصيل سينمائية
            response_format="b64_json"                 # جلب البكسلات بصيغة نصية لفك التشفير
        )

        # 5. استخراج البكسلات والرفع إلى Supabase
        print(f"--- [STEP 4] Extracting and Uploading to Supabase bucket: {BUCKET_NAME} ---")
        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        
        # إنشاء اتصال مع Supabase
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # توليد اسم فريد للملف يعتمد على النمط والوقت الحالي لمنع تداخل أو تكرار الصور
        file_name = f"flux_{style_key}_{int(time.time())}.png"
        
        # رفع الملف بصيغة بايتات مباشرة إلى الباكت
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        # جلب الرابط العام والمباشر للصورة الناتجة لإرسالها للموقع
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] FLUX Image is live at: {public_url} ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "dimensions": f"{width}x{height}",
                "engine": "FLUX.1 [schnell] SDK"
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المستمر والاستماع لطلبات الـ API القادمة من موقعك
runpod.serverless.start({"handler": handler})
