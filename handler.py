# handler.py
# 🚀 تفعيل نظام الأمان وفك تشفير الـ JSON في الذاكرة كأول خطوة عند إقلاع السيرفر
import api_key 

import os# handler.py
# 🚀 تفعيل نظام الأمان وفك تشفير الـ JSON في الذاكرة كأول خطوة عند إقلاع السيرفر
import api_key 

import os
import base64
import time
import runpod
from supabase import create_client

# استدعاء مكتبة Google Cloud الرسمية لـ Vertex AI
from google.cloud import aiplatform

# الاستيراد المباشر من ملفات مشروعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
from dimensions_config import get_image_dimensions

# إعدادات Supabase الخاصة بك
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

# إعدادات بيئة جوجل (يتم قراءتها أو حقنها تلقائياً عبر ملف api_key)
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

        # ⚡️ تهيئة اتصال واجهة Google Vertex AI بالمشروع والمنطقة
        aiplatform.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        
        # تحويل الأبعاد الرقمية إلى نسب الـ Aspect Ratio الذكية لـ Imagen 3
        aspect_ratio = "1:1"
        if width > height: 
            aspect_ratio = "16:9"
        elif height > width: 
            aspect_ratio = "9:16"

        # 🛠️ التعديل الجوهري: تمرير البيانات كـ Dictionary مباشر لضمان الاستقرار التام وتجنب أخطاء الـ SDK
        instances = [{"prompt": final_positive_prompt}]
        parameters = {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
            "outputMimeType": "image/png"
        }

        # إنشاء عميل الاتصال بسيرفر جوجل المحدد ديناميكياً
        client_options = {"api_endpoint": f"{GOOGLE_LOCATION}-aiplatform.googleapis.com"}
        from google.cloud.aiplatform_v1 import PredictionServiceClient
        client = PredictionServiceClient(client_options=client_options)
        
        # بناء المسار النهائي للـ Endpoint الخاص بـ Imagen 3 المستقر
        endpoint = client.endpoint_path(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION, endpoint="imagen-3.0-generate-002")
        
        # استدعاء سيرفرات جوجل السحابية الفائقة لتوليد الصورة
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
        
        # استخراج بيانات الصورة المرتجعة بصيغة Base64 وتحويلها إلى بايتات حقيقية للرفع
        # الهيكلية الجديدة لردود جوجل تعتمد على الحقول الديناميكية القاموسية
        prediction = response.predictions[0]
        b64_data = prediction.get("bytesBase64Encoded")
        
        if not b64_data:
            return {"error": "Failed to get image data from Google Response."}
            
        image_bytes = base64.b64decode(b64_data)

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

import base64
import time
import runpod
from supabase import create_client

# استدعاء مكتبات Google Cloud الرسمية لـ Vertex AI
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

# الاستيراد المباشر من ملفات مشروعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
from dimensions_config import get_image_dimensions

# إعدادات Supabase الخاصة بك
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

# إعدادات بيئة جوجل (يتم قراءتها أو حقنها تلقائياً عبر ملف api_key)
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

        # ⚡️ تهيئة اتصال واجهة Google Vertex AI بالمشروع والمنطقة دقيقة بدقيقة
        aiplatform.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        
        # تحويل الأبعاد الرقمية إلى نسب الـ Aspect Ratio الذكية التي يتطلبها محرك Imagen 3
        # لضمان عدم مط الصورة أو تشويه الأبعاد التكوينية للبورتريه أو اللاندسكيب
        aspect_ratio = "1:1"
        if width > height: 
            aspect_ratio = "16:9"
        elif height > width: 
            aspect_ratio = "9:16"

        # تجهيز الطلب بصيغة الـ Schema الرسمية لجوجل كلاود
        instances = [predict.instance.ImageGenerationPredictionInstance(prompt=final_positive_prompt).to_value()]
        parameters = predict.params.ImageGenerationPredictionParams(
            sample_count=1, 
            aspect_ratio=aspect_ratio,
            output_mime_type="image/png"
        ).to_value()

        # إنشاء عميل الاتصال بسيرفر جوجل المحدد
        client_options = {"api_endpoint": f"{GOOGLE_LOCATION}-aiplatform.googleapis.com"}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        
        # بناء المسار النهائي للـ Endpoint الخاص بـ Imagen 3 المستقر
        endpoint = client.endpoint_path(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION, endpoint="imagen-3.0-generate-002")
        
        # استدعاء سيرفرات جوجل السحابية الفائقة لتوليد الصورة
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
        
        # استخراج بيانات الصورة المرتجعة بصيغة Base64 وتحويلها إلى بايتات حقيقية للرفع
        b64_data = response.predictions[0]["bytesBase64Encoded"]
        image_bytes = base64.b64decode(b64_data)

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

# بدء تشغيل Runpack Serverless الاستماع للطلبات القادمة من موقعك
runpod.serverless.start({"handler": handler})
