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

# القاموس الشامل لكل الستايلات المطلوبة
STYLE_MODIFIERS = {
    "realistic": "photorealistic, ultra-detailed, 8k uhd, raw photo, master part, highly professional",
    "anime": "anime style, studio ghibli, vibrant colors, detailed lineart, high resolution",
    "cinematic": "cinematic lighting, dramatic shadows, movie still, 35mm lens, anamorphic, sharp focus",
    "cartoon": "cartoon style, 2d animation, clean lines, bold colors, playful design",
    "pixar": "pixar animation style, 3d render, disney style, cute character, subsurface scattering, octane render, 4k"
}

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # دمج الستايل المختار
        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
        full_prompt = f"{user_prompt}, {modifier}"

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        image = pipe(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, distorted, low resolution, bad hands, deformed faces",
            num_inference_steps=35,
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
