import runpod
import torch
import base64
from io import BytesIO
import os
from PIL import Image

# حل مشكلات التوافق
if not hasattr(torch, 'xpu'):
    torch.xpu = type('XPU', (), {'is_available': lambda: False, 'empty_cache': lambda: None})

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

MODEL_CACHE_DIR = "/workspace/models"

# قاموس الستايلات لتعزيز الجودة تلقائياً
STYLE_MODIFIERS = {
    "realistic": "photorealistic, ultra-detailed, 8k uhd, Fujifilm XT4, highly professional, raw photo",
    "anime": "anime style, vibrant colors, detailed lineart, high resolution, Studio Ghibli inspired",
    "cinematic": "cinematic lighting, dramatic shadows, movie still, masterpiece, anamorphic lens flare",
    "digital_art": "digital painting, concept art, sharp focus, trending on artstation, intricate details"
}

def handler(job):
    try:
        job_input = job['input']
        
        # استلام المعطيات من المستخدم (مع قيم افتراضية)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)
        upscale = job_input.get('upscale', False) # خيار التكبير لـ Full HD

        # 1. دمج الستايل مع الوصف
        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
        full_prompt = f"{user_prompt}, {modifier}"
        
        print(f"--- [START] Generating Style: {style} | Size: {width}x{height} ---")

        # 2. تحميل الموديل
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        # 3. توليد الصورة
        image = pipe(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, distorted, low resolution, bad anatomy, deformed",
            num_inference_steps=35,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # 4. التكبير الرقمي إذا طلب المستخدم (Upscaling)
        if upscale:
            # التكبير بنسبة 1.5 للوصول لـ Full HD وما فوق
            new_size = (int(image.width * 1.5), int(image.height * 1.5))
            image = image.resize(new_size, Image.LANCZOS)
            print(f"--- Upscaled to: {image.width}x{image.height} ---")

        # 5. التشفير والإرسال
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_base64": img_str,
            "info": {
                "final_size": f"{image.width}x{image.height}",
                "style_used": style
            }
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
