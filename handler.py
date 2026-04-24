import runpod
import torch
import base64
from io import BytesIO
import os
from PIL import Image

# ستايلات إضافية بما فيها الكرتوني
STYLE_MODIFIERS = {
    "cartoon": "cartoon style, Pixar animation, vibrant colors, clean lines, 3d render, cute character design, high resolution",
    "realistic": "photorealistic, ultra-detailed, 8k uhd, Fujifilm XT4, raw photo",
    "cinematic": "cinematic lighting, dramatic shadows, movie still, masterpiece",
    "anime": "anime style, Studio Ghibli, detailed lineart"
}

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

MODEL_CACHE_DIR = "/workspace/models"

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'cartoon') # الافتراضي كرتوني
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["cartoon"])
        full_prompt = f"{user_prompt}, {modifier}"

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        # توليد الصورة (للحصول على تأثير الظهور التدريجي برمجياً، نحتاج لتقليل الخطوات في نسخة أولية)
        # لكن للتبسيط، سنرسل الصورة بجودة تصاعدية عبر الـ Steps
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=30, # عدد خطوات كافٍ للجودة والسرعة
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
