import os
# إيقاف الوظائف التجريبية لضمان استقرار NumPy وحل مشكلة الإصدارات
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import torch
import runpod
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from video_engine import generate_video_logic

# إعداد السجلات لمراقبة العملية
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- إعدادات الوصول ---
# تأكد من وضع المفتاح الصحيح الذي يبدأ بـ sk-proj-
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CACHE_DIR = "/workspace/models"

# قاموس الستايلات الفنية لدمجها مع النص المحسن
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed, sharp focus",
    "anime": "anime style art, vibrant colors, studio ghibli inspired, high quality digital painting",
    "cinematic": "cinematic movie still, moody lighting, epic composition, 35mm lens",
    "cartoon": "cute 3d cartoon style, disney animation, pixar style, vibrant colors",
    "pixar": "high-end 3d render, pixar movie aesthetics, soft global illumination"
}

def enhance_prompt_via_gpt(user_input):
    """
    وظيفة إجبارية لترجمة وتحسين النص باستخدام GPT-4o-mini
    """
    try:
        logger.info(f"Original user input: {user_input}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a professional SDXL prompt engineer. "
                        "Your task is to: 1. Translate the input to English if it's in another language. "
                        "2. Enhance the prompt with vivid, descriptive visual details. "
                        "3. Return ONLY the enhanced English prompt without any preamble or quotes."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            timeout=10
        )
        enhanced = response.choices[0].message.content.strip()
        logger.info(f"Enhanced prompt from GPT: {enhanced}")
        return enhanced
    except Exception as e:
        logger.error(f"OpenAI Enhancement Failed: {e}")
        # في حال الفشل نعود للنص الأصلي لضمان عدم توقف الخدمة
        return user_input

# --- تحميل موديل SDXL ---
logger.info("Loading Stable Diffusion XL Model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

# استخدام Scheduler احترافي وسريع
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'image')

        # 1. معالجة طلبات الفيديو
        if mode == 'video':
            logger.info("Redirecting to Video Engine...")
            return generate_video_logic(job)

        # 2. معالجة طلبات الصور (مع التحسين الإجباري)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        # تنفيذ التحسين والترجمة عبر OpenAI
        enhanced_text = enhance_prompt_via_gpt(user_prompt)
        
        # دمج النص المحسن مع الستايل المختار
        style_details = STYLE_MODIFIERS.get(style, "")
        final_full_prompt = f"{enhanced_text}, {style_details}"
        
        logger.info(f"Final prompt being rendered: {final_full_prompt}")

        with torch.inference_mode():
            image = pipe(
                prompt=final_full_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024),
                guidance_scale=7.5
            ).images[0]

        # تحويل الصورة إلى Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": image_base64}

    except Exception as e:
        logger.error(f"Handler Error: {e}")
        return {"error": str(e)}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
