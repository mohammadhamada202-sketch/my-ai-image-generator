# handler.py
import os
import base64
import time
import requests
import runpod
from supabase import create_client

# استدعاء المكتبة الرسمية لـ Together
from together import Together

# الاستيراد الدقيق والنظامي لملفات الإعدادات والمساعدين من مستودعك
from translator_helper import get_epic_prompt
from styles_config import STYLE_CONFIGS
from dimensions_config import get_image_dimensions  # استدعاء دالتك الاحترافية التي أرسلتها بالملي

# جلب متغيرات البيئة من RunPod
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()  # مفتاح هقينج فيس الجديد للكرتون المجاني
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BUCKET_NAME = "MyFirstImagesTest1"

def handler(job):
    try:
        print("--- [START] SMARTGEN CLEAN HYBRID ENGINE ---")
        job_input = job.get('input', {})
        
        raw_prompt = job_input.get('prompt')
        style_key = job_input.get('style', 'photorealistic')

        if not raw_prompt:
            print("--- [ERROR] Input prompt is missing ---")
            return {"error": "Prompt is missing."}

        # 1. استدعاء المترجم والمحسن الذكي (GPT-4o)
        print("--- [STEP 1] Activating OpenAI Prompt Engineer (GPT-4o)... ---")
        optimized_prompt = get_epic_prompt(raw_prompt)

        # 2. جلب إعدادات النمط والموديل المخصص له ديناميكياً من ملف الستايلات الخاص بك
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["photorealistic"])
        
        provider = config["provider"]
        target_model = config["model"]
        style_enhancer = config["prompt_enhancer"]
        
        final_positive_prompt = f"{optimized_prompt}, {style_enhancer}"

        # 3. استدعاء دالتك الاحترافية لجلب الأبعاد والمقاسات الديناميكية
        width, height = get_image_dimensions(job_input)
        print(f"--- [STEP 2] Routed -> Style: {style_key} | Provider: {provider} | Model: {target_model} | Resolution: {width}x{height} ---")

        image_bytes = None

        # 4. التوجيه الذكي (شرطي المرور) بناءً على الـ Provider المحدد داخل ملف الستايلات
        if provider == "huggingface":
            print(f"--- [STEP 3] Generating via Hugging Face Serverless API (Free 2D)... ---")
            HF_API_URL = f"https://api-inference.huggingface.co/models/{target_model}"
            
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": final_positive_prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": 28  # خطوات كافية لجودة الكرتون الـ 2D الصافية
                    }
                }
            )
            
            if response.status_code != 200:
                print(f"--- [FAILED] Hugging Face Error: {response.text} ---")
                return {"error": f"Hugging Face API Error: {response.text}"}
                
            image_bytes = response.content

        elif provider == "together":
            print(f"--- [STEP 3] Generating via Together Official SDK (FLUX Engine)... ---")
            client = Together()
            response = client.images.generate(
                model=target_model,
                prompt=final_positive_prompt,
                width=width,
                height=height,
                steps=4,  # موديل فلوكس شنيل يحتاج 4 خطوات فقط
                response_format="b64_json"
            )
            
            b64_data = response.data[0].b64_json
            image_bytes = base64.b64decode(b64_data)
            
        else:
            return {"error": f"Unknown provider configured for style: {style_key}"}

        # 5. الرفع المشترك والمستقر إلى باكت Supabase
        print(f"--- [STEP 4] Uploading Output to Supabase bucket: {BUCKET_NAME} ---")
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # تسمية فريدة للملف تمنع التداخل والتكرار في الباكت
        file_name = f"smartgen_{style_key}_{int(time.time())}.png"
        
        storage = sb_client.storage.from_(BUCKET_NAME)
        storage.upload(path=file_name, file=image_bytes, file_options={"content-type": "image/png"})
        
        public_url = storage.get_public_url(file_name)
        print(f"--- [SUCCESS] Engine processing complete. Image live! ---")
        
        return {
            "image_url": public_url,
            "status": "success",
            "metadata": {
                "style_used": style_key,
                "dimensions": f"{width}x{height}",
                "provider_used": provider,
                "model_used": target_model
            }
        }

    except Exception as e:
        print(f"--- [FATAL ERROR] ---: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر المستمر على RunPod
runpod.serverless.start({"handler": handler})
