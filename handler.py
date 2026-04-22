import runpod
import torch
import io
import base64
import os
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# 1. إعداد السيرفر وتحميل الموديل في الذاكرة (خارج الـ handler لتسريع الطلبات اللاحقة)
def load_model():
    print("--- [1/4] جاري تجهيز كرت الشاشة وتحميل الموديل... ---")
    model_id = "SG161222/RealVisXL_V4.0"
    
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # تحسين محرك الرسم للحصول على دقة واقعية
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("--- [2/4] تم تحميل الموديل بنجاح ووضعه في الـ VRAM ---")
        return pipe
    except Exception as e:
        print(f"❌ خطأ فادح في تحميل الموديل: {str(e)}")
        return None
        

# تشغيل التحميل الأولي
generator_pipe = load_model()

def handler(job):
    try:
        # الحصول على البيانات من الطلب
        job_input = job['input']
        prompt = job_input.get('prompt', 'A high quality professional photo')
        
        print(f"--- [3/4] استلمت طلباً جديداً: {prompt} ---")

        if generator_pipe is None:
            return {"error": "الموديل لم يتم تحميله بشكل صحيح في السيرفر"}

        # عملية الرسم الاحترافية
        with torch.inference_mode():
            image = generator_pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted, watermark, text, bad anatomy",
                num_inference_steps=25,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]

        # تحويل الصورة إلى Base64 لإرسالها للموقع
        print("--- [4/4] تم الرسم بنجاح! جاري تحويل البيانات... ---")
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=90)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {"image": image_base64}

    except Exception as e:
        print(f"❌ حدث خطأ أثناء المعالجة: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر
runpod.serverless.start({"handler": handler})
