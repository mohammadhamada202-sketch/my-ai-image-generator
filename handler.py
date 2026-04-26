import os
# إجبار النظام على رؤية المكتبات بشكل صحيح
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import runpod
import torch
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from video_engine import generate_video_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- الإعدادات ---
OPENAI_API_KEY = "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" 
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_CACHE_DIR = "/workspace/models"

STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed",
    "anime": "anime style art, vibrant colors, studio ghibli inspired",
    "cinematic": "cinematic movie still, moody lighting, epic composition",
    "cartoon": "cute 3d cartoon style, disney animation, pixar style",
    "pixar": "high-end 3d render, pixar movie aesthetics, masterpiece"
}

def enhance_prompt_via_gpt(user_input):
    if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY: return user_input
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Convert user input into a detailed English visual prompt for SDXL. Return only the prompt."}],
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return user_input

# تحميل الموديل
logger.info("Loading SDXL Model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'image')

        if mode == 'video':
            return generate_video_logic(job)

        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        enhanced = enhance_prompt_via_gpt(user_prompt)
        final_prompt = f"{enhanced}, {STYLE_MODIFIERS.get(style, '')}"
        
        logger.info(f"Generating Image: {final_prompt}")

        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
