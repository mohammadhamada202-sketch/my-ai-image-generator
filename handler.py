import os
import runpod
import torch
import io
import base64
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعدادات لضمان السرعة والدقة
os.environ["TRANSFORMERS_OFFLINE"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SG161222/RealVisXL_V4.0"

# تحميل الموديل بأعلى دقة ممكنة (FP16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to(device)

# محرك الرسم الاحترافي (لصور Full HD واقعية)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', 'A high quality photo')
        
        # إعدادات الدقة العالية (Full HD)
        # العرض 1024 والطول 1024 هو المعيار الذهبي لـ SDXL
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted, watermark, text",
                num_inference_steps=30, # عدد خطوات كافي لدقة عالية جداً
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]
        
        # حفظ الصورة بأقل فقد في الجودة
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=95)
        return {"image": base64.b64encode(buf.getvalue()).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
