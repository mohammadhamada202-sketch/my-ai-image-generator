import runpod
import torch
import base64
from io import BytesIO
import os
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# المسار الخاص بالـ Network Volume لتخزين الموديل
MODEL_CACHE_DIR = "/workspace/models"

# قاموس الستايلات الفنية المدعومة
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k uhd, raw photo, ultra-detailed, highly professional, masterpiece",
    "anime": "anime style, studio ghibli, vibrant colors, detailed lineart, high resolution",
    "cinematic": "cinematic lighting, dramatic shadows, movie still, 35mm lens, sharp focus",
    "cartoon": "cartoon style, 2d animation, clean lines, bold colors, playful design",
    "pixar": "pixar animation style, 3d render, disney style, cute character, subsurface scattering, 4k"
}

def handler(job):
    try:
        job_input = job['input']
        
        # استلام المعطيات من الموقع
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # تجهيز الوصف النهائي بناءً على الستايل المختبر
        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
        full_prompt = f"{user_prompt}, {modifier}"
        
        # تحميل الموديل RealVisXL V4.0
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        # تحسين الأداء وسرعة التوليد
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass # في حال عدم توفر xformers سيعمل الكود بشكل طبيعي

        # توليد الصورة
        image = pipe(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, distorted, low resolution, bad hands, deformed faces, extra fingers",
            num_inference_steps=35,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # تحويل الصورة إلى صيغة Base64 لإرسالها للواجهة الأمامية
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        return {"error": str(e)}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
