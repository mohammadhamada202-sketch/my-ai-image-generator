import numpy as np  # استدعاء نيمباي في أول سطر لحل مشكلة RuntimeError
import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from video_engine import generate_video_logic

# إعدادات السجلات لمراقبة أداء السيرفر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- إعدادات الوصول ---
# ضع مفتاح OpenAI الخاص بك هنا
OPENAI_API_KEY = "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" 
client = OpenAI(api_key=OPENAI_API_KEY)

# مسار تخزين الموديلات في الفوليوم لضمان السرعة
MODEL_CACHE_DIR = "/workspace/models"

# قاموس الستايلات الفنية (يتم دمجها مع البرومبت المحسن)
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed, sharp focus, f/1.8",
    "anime": "anime style art, vibrant colors, studio ghibli inspired, high quality digital painting",
    "cinematic": "cinematic movie still, moody lighting, epic composition, 35mm lens, anamorphic lens flares",
    "cartoon": "cute 3d cartoon style, disney animation, pixar style, vibrant colors, smooth textures",
    "pixar": "high-end 3d render, pixar movie aesthetics, soft global illumination, masterpiece digital art"
}

def enhance_prompt_via_gpt(user_input):
    """
    وظيفة ترجمة وتحسين النص باستخدام GPT-4o-mini
    تحول المدخلات من أي لغة إلى برومبت احترافي بالإنجليزية
    """
    if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
        return user_input
    
    try:
        logger.info(f"Original Prompt: {user_input}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer for SDXL. Convert the user input into a detailed, descriptive English visual prompt. Return only the final prompt without any introduction."},
                {"role": "user", "content": user_input}
            ],
            timeout=10
        )
        enhanced = response.choices[0].message.content.strip()
        logger.info(f"Enhanced Prompt: {enhanced}")
        return enhanced
    except Exception as e:
        logger.error(f"OpenAI Enhancement Failed: {e}")
        return user_input # في حال فشل GPT نستخدم النص الأصلي لضمان استمرار العمل

# --- تحميل موديل SDXL ---
# تحميل الموديل مرة واحدة عند بدء تشغيل الـ Pod لتوفير الوقت
logger.info("Loading Stable Diffusion XL Model...")
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE_DIR
    ).to("cuda")

    # استخدام Scheduler سريع واحترافي
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    logger.info("Model loaded successfully onto GPU.")
except Exception as e:
    logger.error(f"Critical Error loading model: {e}")

def handler(job):
    """
    الدالة الرئيسية التي تستقبل طلبات RunPod
    """
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'image') # التمييز بين وضع الصورة والفيديو

        # 1. معالجة طلبات الفيديو
        if mode == 'video':
            logger.info("Redirecting request to Video Engine...")
            return generate_video_logic(job)

        # 2. معالجة طلبات الصور (الوضع الافتراضي)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # تحسين النص وترجمته
        final_prompt = enhance_prompt_via_gpt(user_prompt)
        # إضافة لمسات الستايل
        full_style_prompt = f"{final_prompt}, {STYLE_MODIFIERS.get(style, '')}"
        
        logger.info("Starting Image Generation...")
        
        # عملية التوليد الفعلية
        with torch.inference_mode():
            image = pipe(
                prompt=full_style_prompt,
                num_inference_steps=30,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]

        # تحويل النتيجة إلى Base64 لإرسالها للـ HTML
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info("Generation complete.")
        return {"image_base64": image_str}

    except Exception as e:
        logger.error(f"Handler processing error: {e}")
        return {"error": str(e)}

# تشغيل العامل (Worker)
runpod.serverless.start({"handler": handler})
