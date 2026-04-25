import runpod
import torch
import base64
import os
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
# استيراد محرك الفيديو من الملف الثاني
from video_engine import generate_video_logic

# إعداد السجلات (Logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# ضع مفتاح OpenAI الخاص بك هنا مباشرة بين علامات التنصيص
# ---------------------------------------------------------
OPENAI_API_KEY = "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" 

# تعريف مكتبة OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CACHE_DIR = "/workspace/models"

# إضافات الستايلات لتحسين النتائج
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed, f/1.8",
    "anime": "anime art, studio ghibli style, vibrant digital painting, high resolution",
    "cinematic": "cinematic shot, moody lighting, epic composition, 35mm lens, depth of field",
    "cartoon": "cute 3d cartoon style, disney animation, vibrant colors, smooth textures",
    "pixar": "pixar movie style, 3d render, high quality digital art, soft lighting"
}

def enhance_prompt_via_gpt(user_input):
    """وظيفة لتحسين وترجمة النص عبر GPT"""
    if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
        return user_input
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Convert the user input into a detailed English visual prompt for SDXL. Return only the prompt."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        enhanced_text = response.choices[0].message.content.strip()
        logger.info(f"Enhanced Prompt: {enhanced_text}")
        return enhanced_text
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return user_input # العودة للنص الأصلي في حال فشل OpenAI

# تحميل موديل الصور SDXL عند بدء تشغيل السيرفر
logger.info("Loading SDXL Model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", # موديل واقعي جداً
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

# تحسين سرعة الموديل
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        mode = job_input.get('mode', 'image') # افتراضياً وضع الصور

        # 1. إذا كان الطلب فيديو (تحويل صورة لفيديو)
        if mode == 'video':
            logger.info("Mode: Video - Redirecting to Video Engine...")
            return generate_video_logic(job)

        # 2. إذا كان الطلب صورة (إنشاء صورة من نص)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # تحسين النص عبر GPT
        enhanced_prompt = enhance_prompt_via_gpt(user_prompt)
        
        # دمج النص مع إضافات الستايل المختارة
        final_prompt = f"{enhanced_prompt}, {STYLE_MODIFIERS.get(style, '')}"
        
        logger.info(f"Generating Image for: {final_prompt}")

        # عملية التوليد
        image = pipe(
            prompt=final_prompt,
            num_inference_steps=30,
            width=width,
            height=height
        ).images[0]

        # تحويل الصورة إلى Base64 لإرسالها للمتصفح
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": image_base64}

    except Exception as e:
        logger.error(f"Handler Error: {e}")
        return {"error": str(e)}

# بدء تشغيل سيرفر RunPod
runpod.serverless.start({"handler": handler})
