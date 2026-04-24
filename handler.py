import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد الـ Logging لرؤية التفاصيل في RunPod Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد مفتاح OpenAI (تأكد من وضعه في Environment Variables في RunPod)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CACHE_DIR = "/workspace/models"

# مصفوفة الستايلات لنتائج احترافية
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k uhd, masterpiece, raw photo, highly detailed, soft cinematic lighting",
    "anime": "anime style, vibrant, high-quality digital art, sharp focus, studio ghibli aesthetic",
    "cinematic": "cinematic movie still, moody lighting, anamorphic lens flare, 35mm, high contrast",
    "pixar": "disney pixar 3d style, cute, highly detailed render, subsurface scattering, vibrant colors"
}

def enhance_prompt_via_gpt(user_input):
    """وظيفة ذكية لفهم العامية وترجمتها وتحسينها"""
    logger.info(f"--- Starting GPT translation for: {user_input} ---")
    try:
        instruction = (
            "You are a professional Prompt Engineer for SDXL. "
            "Convert the user's input (it could be Arabic slang, formal, or any language) "
            "into a very descriptive, high-quality English visual prompt. "
            "Focus on lighting, textures, and artistic style. "
            "Output ONLY the final English prompt text."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        enhanced_text = response.choices[0].message.content.strip()
        logger.info(f"--- GPT Enhanced Prompt: {enhanced_text} ---")
        return enhanced_text
    except Exception as e:
        logger.error(f"!!! OpenAI Error: {str(e)} !!!")
        return user_input  # في حال الفشل نستخدم النص الأصلي لكي لا يتوقف البرنامج

# تحميل الموديل (خارج الـ handler لتسريع التشغيل)
logger.info("Loading SDXL Model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
logger.info("Model Loaded Successfully!")

def handler(job):
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # 1. الترجمة والتحسين (GPT)
        final_description = enhance_prompt_via_gpt(user_prompt)
        
        # 2. إضافة لمسات الستايل المختارة
        modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
        full_prompt = f"{final_description}, {modifier}"
        
        logger.info(f"Generating image with full prompt: {full_prompt}")

        # 3. توليد الصورة
        image = pipe(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, bad anatomy, deformed, watermark, text, grainy",
            num_inference_steps=30,
            guidance_scale=7.0,
            width=width,
            height=height
        ).images[0]

        # 4. تحويل الصورة إلى Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        return {"error": str(e)}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
