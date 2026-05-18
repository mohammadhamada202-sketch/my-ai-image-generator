# handler.py
# 🚀 تفعيل نظام الأمان وفك تشفير الـ JSON في الذاكرة أولاً
import api_key 

import os
import base64
import time
import runpod
from supabase import create_client

# استدعاء مكتبة Google Cloud الرسمية لـ Vertex AI
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# الاستيراد المباشر من ملفات مشروعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS, AVATAR_NEGATIVE_PROMPT
from dimensions_config import get_image_dimensions

# إعدادات Supabase الخاصة بك
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

# إعدادات بيئة جوجل
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "hip-gecko-496121-f9").strip()
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1").strip()

def handler(job):
    try:
        print("--- [START] SMARTGEN ALL-GOOGLE IMAGEN 3 ENGINE ---")
        job_input = job.get('input', {})
        
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            return {"error": "Prompt is missing."}

        # جلب إعدادات النمط والموديل ديناميكياً
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]

        # 1️⃣ تشغيل المترجم والمحسن (OpenAI Prompt Engineer) كخطوة أساسية أولى لكل شيء
        print("--- [INFO] Activating OpenAI Prompt Engineer & Translator ---")
        try:
            # ترجمة وتحسين النص الأصلي (سواء عربي أو إنجليزي) ليعطينا وصف سينمائي فخم
            optimized_prompt = get_epic_prompt(raw_prompt)
            print(f"--- [SUCCESS] Prompt Optimized to: '{optimized_prompt}' ---")
        except Exception as trans_err:
            print(f"--- [WARNING] Translator failed: {str(trans_err)} | Using raw prompt ---")
            optimized_prompt = raw_prompt

        # 2️⃣ قائمة ذكية بالشخصيات المشهورة (بالإنجليزي لأن النص تترجم هلق!)
        famous_characters = ["tom and jerry", "tom & jerry", "mickey mouse", "spiderman", "superman", "batman", "popeye", "pikachu", "goku"]
        
        # الفحص يتم الآن على النص المترجم الفخم أو النص الأصلي للاحتياط
        is_famous = any(char in optimized_prompt.lower() for char in famous_characters) or any(char in raw_prompt.lower() for char in famous_characters)

        # 3️⃣ بناء البرومبت النهائي الموجه لجوجل بناءً على الستايل والحماية
        if style_key == "cartoon" or is_famous:
            # لو كرتون أو شخصية مشهورة، بنضيف غلاف الـ Fan-art فوق النص المترجم الجاهز لحمايته من الفلاتر
            print(f"--- [INFO] Famous Character or Cartoon detected | Applying Safety Bypass to Optimized Prompt ---")
            final_positive_prompt = f"stylized fan-art character artistic illustration of {optimized_prompt}, {style_enhancer}"
        else:
            # لو ستايل عادي (واقعي، أنمي.. إلخ)، بيمرق النص المترجم والمحسن مع محسن الستايل الطبيعي
            print("--- [INFO] Standard Style: Applying standard prompt combination ---")
            final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # المقاسات والأبعاد
        width, height = get_image_dimensions(job_input)
        print(f"--- [ROUTE] Style: {style_key} | Engine: GOOGLE VERTEX AI ({target_model}) ---")

        # تهيئة واجهة Google Vertex AI
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        
        # تحويل الأبعاد لنسب ذكية لـ Imagen 3
        aspect_ratio = "1:1"
        if width > height: 
            aspect_ratio = "16:9"
        elif height > width: 
            aspect_ratio = "9:16"

        # تحميل موديل جوجل
        print(f"--- [CALL] Loading Google Model: {target_model} ---")
        model = ImageGenerationModel.from_pretrained(target_model)
        
        # استدعاء توليد الصورة مع إعدادات حماية مرنة لتقليل الحظر العشوائي للـ Fan Art
        response = model.generate_images(
            prompt=final_positive_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="block_few"
        )
        
        # الفحص الحاسم لمنع خطأ list index out of range في حال تم حظر المحتوى كلياً
        if not response.images:
            print("--- [SAFETY BLOCKED] Google Vertex AI blocked or refused to generate this image ---")
            return {"error": "The prompt was flagged or blocked by Google Safety Filters. Please try a different description."}
            
        # استخراج بايتات الصورة بأمان
        image_bytes = response.images[0]._image_bytes

        # 📦 الرفع إلى باكت Supabase
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
                "aspect_ratio_used": aspect_ratio,
                "provider": "google"
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
