import runpod
import torch
import io
import base64
import os
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعدادات لضمان عدم حدوث تعارض مع معالجات Intel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model():
    print("--- [1/4] جاري تجهيز كرت الشاشة NVIDIA وتحميل الموديل... ---")
    model_id = "SG161222/RealVisXL_V4.0"
    
    try:
        # تحميل الموديل مع تحديد الجهاز cuda يدوياً لتجنب خطأ xpu
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # تحسين محرك الرسم
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("--- [2/4] تم تحميل الموديل بنجاح على RTX GPU ---")
        return pipe
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {str(e)}")
        return None

# تشغيل التحميل الأولي عند إقلاع السيرفر
generator_pipe = load_model()

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', 'A high quality professional photo')
        
        print(f"--- [3/4] استلمت طلباً للرسم: {prompt} ---")

        if generator_pipe is None:
            return {"error": "الموديل لم يتم تحميله بشكل صحيح في السيرفر"}

        # عملية الرسم
        with torch.inference_mode():
            image = generator_pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted, watermark, text, bad anatomy",
                num_inference_steps=25,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]

        # تحويل الصورة إلى Base64
        print("--- [4/4] اكتمل الرسم! جاري إرسال النتيجة... ---")
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=90)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {"image": image_base64}

    except Exception as e:
        print(f"❌ حدث خطأ أثناء الرسم: {str(e)}")
        return {"error": str(e)}

# تشغيل محرك RunPod
runpod.serverless.start({"handler": handler})
