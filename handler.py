import runpod
import torch
import base64
from io import BytesIO
import os

# --- حل مشكلة تعارض المكتبات (torch.xpu) ---
if not hasattr(torch, 'xpu'):
    torch.xpu = type('XPU', (), {'is_available': lambda: False, 'empty_cache': lambda: None})

# استيراد مكتبات diffusers بعد حل مشكلة torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# المسار المربوط بالـ Network Volume (100GB)
MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        # 1. التأكد من وجود مجلد التخزين الدائم
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            print(f"--- تم إنشاء مجلد التخزين: {MODEL_CACHE_DIR} ---")

        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        if not prompt:
            return {"error": "الرجاء إدخال وصف للصورة (Prompt)"}

        print(f"--- [START] جاري معالجة الطلب: {prompt} ---")

        # 2. تحميل الموديل (سيتم حفظه في الـ Volume للأبد)
        print(f"--- فحص الموديل في المجلد الدائم: {MODEL_CACHE_DIR} ---")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        # إعداد المحرك للحصول على أفضل جودة
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # تحسين استهلاك الذاكرة (Memory Optimization)
        pipe.enable_xformers_memory_efficient_attention()

        # 3. عملية توليد الصورة
        print("--- بدأت عملية الرسم... ---")
        image = pipe(
            prompt=prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 4. تحويل النتيجة إلى Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("--- [DONE] تم توليد الصورة بنجاح وتخزينها في الـ Volume ---")
        return {"image_base64": img_str}

    except Exception as e:
        error_msg = f"--- [ERROR]: {str(e)} ---"
        print(error_msg)
        return {"error": str(e)}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
