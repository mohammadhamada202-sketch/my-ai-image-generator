import runpod
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import base64
from io import BytesIO
import os

# المسار الذي قمت بربطه في الـ Network Volume
# تأكد أن Mount Path في إعدادات RunPod هو /workspace/models
MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        # 1. التأكد من وجود المجلد في الـ Volume
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            print(f"--- تم إنشاء المجلد: {MODEL_CACHE_DIR} ---")

        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        if not prompt:
            return {"error": "الرجاء إدخال وصف للصورة (Prompt)"}

        print(f"--- [START] جاري العمل على الطلب: {prompt} ---")

        # 2. تحميل الموديل (سيتم تحميله من الإنترنت في أول مرة فقط وتخزينه في الـ Volume)
        print(f"--- فحص الموديل في: {MODEL_CACHE_DIR} ---")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE_DIR # هذا السطر هو مفتاح التوفير والسرعة
        ).to("cuda")
        
        # إعداد المحرك (Scheduler) ليعطي أفضل جودة
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # تحسين الذاكرة لضمان عدم حدوث Crash
        pipe.enable_xformers_memory_efficient_attention()

        # 3. عملية الرسم (Inference)
        print("--- بدأت عملية الرسم الآن... ---")
        image = pipe(
            prompt=prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 4. تحويل الصورة إلى نص (Base64) لإرسالها لموقعك
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("--- [DONE] تم توليد الصورة بنجاح ---")
        return {"image_base64": img_str}

    except Exception as e:
        error_msg = f"--- [ERROR]: {str(e)} ---"
        print(error_msg)
        return {"error": str(e)}

# تشغيل السيرفر لاستقبال الطلبات
runpod.serverless.start({"handler": handler})
