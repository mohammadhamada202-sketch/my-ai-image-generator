import runpod
import torch
import base64
from io import BytesIO
import os

# --- 1. حل مشكلات تعارض المكتبات (Compatibility Layer) ---

# حل مشكلة AttributeError: module 'torch' has no attribute 'xpu'
if not hasattr(torch, 'xpu'):
    torch.xpu = type('XPU', (), {'is_available': lambda: False, 'empty_cache': lambda: None})

# حل مشكلة ImportError: cannot import name 'cached_download'
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    import huggingface_hub.file_download
    huggingface_hub.cached_download = huggingface_hub.file_download.hf_hub_download

# الآن يمكن استيراد مكتبات diffusers بأمان
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# --- 2. إعدادات المسارات ---
# تأكد أن Mount Path في RunPod هو /workspace/models
MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        # التأكد من وجود مجلد التخزين الدائم في الـ Volume
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            print(f"--- تم تجهيز مجلد التخزين: {MODEL_CACHE_DIR} ---")

        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        if not prompt:
            return {"error": "يرجى إدخال وصف للصورة (Prompt)"}

        print(f"--- [START] جاري معالجة الطلب: {prompt} ---")

        # 3. تحميل الموديل (تنزيل لمرة واحدة فقط وتخزين دائم)
        print(f"--- فحص الموديل في المجلد الدائم: {MODEL_CACHE_DIR} ---")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        # تحسين المحرك وسرعة الاستجابة
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        # 4. توليد الصورة
        print("--- بدأت عملية الرسم الآن... ---")
        image = pipe(
            prompt=prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 5. تحويل الصورة إلى Base64 لإرسالها للموقع
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
