# handler.py
# 🚀 تفعيل نظام الأمان وفك تشفير الـ JSON في الذاكرة كأول خطوة عند إقلاع السيرفر
import api_key 

import os
import base64
import time
import runpod
from supabase import create_client

# استدعاء مكتبة Google Cloud الرسمية لـ Vertex AI
import vertexai
# الاستدعاء المتوافق والمستقر لكلاس توليد الصور من المسار الحصري للـ Preview
from vertexai.preview.vision_models import ImageGenerationModel

# الاستيراد المباشر من ملفات مشروعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
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

        # جلب إعدادات النمط والموديل ديناميكياً من ملف styles_config.py الخاص بك
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]

        # 🔥 معالجة البرومبت الذكية الخاصة بك بناءً على الستايل
        if style_key == "cartoon":
            print("--- [INFO] Cartoon detected: Passing raw prompt directly to prevent anime distortion ---")
            final_positive_prompt = f"{raw_prompt}, {style_enhancer}"
        else:
            print("--- [INFO] Standard Style: Activating OpenAI Prompt Engineer ---")
            optimized_prompt = get_epic_prompt(raw_prompt)
            final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # المقاسات والأبعاد القادمة من الواجهة الأمامية لموقعك
        width, height = get_image_dimensions(job_input)
        print(f"--- [ROUTE] Style: {style_key} | Engine: GOOGLE VERTEX AI ({target_model}) ---")

        # ⚡️ تهيئة واجهة Google Vertex AI بالمشروع والمنطقة
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        
        # تحويل الأبعاد الرقمية إلى نسب الـ Aspect Ratio الذكية لـ Imagen 3
        aspect_ratio = "1:1"
        if width > height: 
            aspect_ratio = "16:9"
        elif height > width: 
            aspect_ratio = "9:16"

        # 🛠️ استدعاء الموديل من خلال المسار المحدث والمستقر
        print(f"--- [CALL] Loading Google Image Generation Model: {target_model} ---")
        model = ImageGenerationModel.from_pretrained(target_model)
        
        # إرسال طلب التوليد الفوري
        response = model.generate_images(
            prompt=final_positive_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            output_mime_type="image/png"
        )
        
        # استخراج البايتات مباشرة من أول صورة مرتجعة
        image_bytes = response.images[0]._image_bytes

        # 📦 الرفع المشترك إلى باكت Supabase الخاص بك
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

# بدء تشغيل Runpod Serverless الاستماع للطلبات القادمة من موقعك
runpod.serverless.start({"handler": handler})
