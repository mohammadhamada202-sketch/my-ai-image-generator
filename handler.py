import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد الـ Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# جلب المفتاح من متغيرات البيئة
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CACHE_DIR = "/workspace/models"

STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k uhd, masterpiece, raw photo, highly detailed",
    "anime": "anime style, vibrant, high-quality digital art, studio ghibli aesthetic",
    "cinematic": "cinematic movie still, moody lighting, 35mm, high contrast",
    "pixar": "disney pixar 3d style, ultra-detailed render, subsurface scattering"
}

def enhance_prompt_via_gpt(user_input):
    """تحسين البرومبت وفهم اللغات المختلفة"""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API Key is missing!")
        return user_input

    try:
        instruction = (
            "You are a professional Prompt Engineer. Translate the user input to a detailed "
            "English visual prompt for SDXL. Output ONLY the final prompt text."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT Error: {str(e)}")
        return user_input

# تحميل الموديل
logger.info("Loading Model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # 1. الترجمة والتحسين
        final_description = enhance_prompt_via_gpt(user_prompt)
        
        # 2. إضافة الستايل
        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
        full_prompt = f"{final_description}, {modifier}"
        
        # 3. التوليد
        image = pipe(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, bad anatomy, text, watermark",
            num_inference_steps=30,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # 4. التحويل لـ Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
