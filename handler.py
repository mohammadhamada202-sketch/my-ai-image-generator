import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from video_engine import generate_video_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ضع مفتاحك هنا ---
OPENAI_API_KEY = "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" 
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CACHE_DIR = "/workspace/models"

STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed, f/1.8",
    "anime": "anime art, studio ghibli style, vibrant digital painting",
    "cinematic": "cinematic shot, moody lighting, epic composition, 35mm lens",
    "cartoon": "3d cartoon style, disney animation, vibrant",
    "pixar": "pixar movie style, 3d render, soft lighting"
}

def enhance_prompt_via_gpt(user_input):
    try:
        logger.info(f"Enhancing prompt: {user_input}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Convert user input into a detailed English visual prompt for SDXL. Return only the prompt."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return user_input

# تحميل الموديل مع تحديد التوافقية
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
        
        # تحسين النص
        enhanced_prompt = enhance_prompt_via_gpt(user_prompt)
        final_prompt = f"{enhanced_prompt}, {STYLE_MODIFIERS.get(style, '')}"
        
        logger.info(f"Generating on GPU: {final_prompt}")

        # عملية التوليد (هنا كان يحدث الخطأ وتم حله بالـ Dockerfile)
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
        logger.error(f"Handler Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
