import runpod
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from googletrans import Translator
import base64
from io import BytesIO
import os

# إعداد المترجم
translator = Translator()

# المسار داخل الـ Network Volume الذي ربطته
MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        # التأكد من وجود المجلد في الـ Volume
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        print(f"--- [START] جاري العمل على الطلب: {prompt} ---")

        # 1. الترجمة التلقائية
        try:
            detected = translator.detect(prompt)
            if detected.lang != 'en':
                prompt = translator.translate(prompt, dest='en').text
                print(f"--- النص بعد الترجمة: {prompt} ---")
        except Exception as e:
            print(f"--- فشلت الترجمة، استخدام النص الأصلي: {str(e)} ---")

        # 2. تحميل الموديل (سيتم حفظه في الـ Volume للأبد)
        print(f"--- فحص الموديل في: {MODEL_CACHE_DIR} ---")
        
        # ملاحظة: إذا كان المجلد فارغاً سيحمل الآن، إذا كان ممتلئاً سيعمل فوراً
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # تحسين الذاكرة لزيادة السرعة
        pipe.enable_xformers_memory_efficient_attention()

        # 3. توليد الصورة
        print("--- بدأت عملية الرسم الآن... ---")
        image = pipe(
            prompt=prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 4. تحويل الصورة إلى Base64 لإرسالها للموقع
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("--- [DONE] تم توليد الصورة بنجاح وتخزين الموديل في الـ Volume ---")
        return {"image_base64": img_str}

    except Exception as e:
        error_msg = f"--- [ERROR]: {str(e)} ---"
        print(error_msg)
        return {"error": str(e)}

# تشغيل الـ Worker
runpod.serverless.start({"handler": handler})
