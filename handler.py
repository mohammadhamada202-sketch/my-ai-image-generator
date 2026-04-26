import os
# إيقاف وظائف NumPy التجريبية لضمان الاستقرار التام مع PyTorch
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import torch
import runpod
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد السجلات (Logs) لمراقبة عملية الترجمة والتوليد
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- إعدادات الوصول ---
# تأكد من استبدال المفتاح بمفتاح OpenAI الخاص بك (sk-proj-...)
OPENAI_API_KEY = "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" 
client = OpenAI(api_key=OPENAI_API_KEY)

# مسار تخزين الموديل في RunPod
MODEL_CACHE_DIR = "/workspace/models"

# قاموس الستايلات لضمان دقة النتائج مع النص المترجم
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece, highly detailed, sharp focus",
    "anime": "anime style art, vibrant colors, studio ghibli inspired, high quality digital painting",
    "cinematic": "cinematic movie still, moody lighting, epic composition, 35mm lens",
    "cartoon": "cute 3d cartoon style, disney animation, pixar style, vibrant colors",
    "pixar": "high-end 3d render, pixar movie aesthetics, soft global illumination"
}

def translate_and_enhance(user_input):
    """
    تقوم هذه الدالة بإرسال النص لـ OpenAI لترجمته وتحويله لبرومبت احترافي
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
                        "Step 1: Translate the input to English if it's not in English. "
                        "Step 2: Expand the input into a detailed, vivid visual prompt. "
                        "Step 3: Return ONLY the final English prompt text."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            timeout=10
        )
        enhanced_text = response.choices[0].message.content.strip()
        logger.info(f"Enhanced English Prompt: {enhanced_text}")
        return enhanced_text
    except Exception as e:
        logger.error(f"OpenAI translation failed: {e}")
        return user_input # العودة للنص الأصلي في حال فشل الاتصال

# --- تحميل موديل SDXL ---
logger.info("Loading SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

# استخدام Scheduler سريع واحترافي لتقليل وقت التوليد
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    try:
        job_input = job['input']
        
        # 1. جلب المدخلات من موقعك (HTML)
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # 2. الخطوة الحاسمة: ترجمة وتحسين النص العربي/الإنجليزي عبر GPT
        final_english_prompt = translate_and_enhance(user_prompt)
        
        # 3. دمج النص المترجم مع محسنات الستايل المختار
        style_details = STYLE_MODIFIERS.get(style, "")
        full_render_prompt = f"{final_english_prompt}, {style_details}"
        
        logger.info(f"Rendering image for: {full_render_prompt}")

        # 4. عملية التوليد البرمجية
        with torch.inference_mode():
            image = pipe(
                prompt=full_render_prompt,
                num_inference_steps=30,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]

        # 5. تحويل النتيجة لـ Base64 لإرسالها للموقع
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": image_base64}

    except Exception as e:
        logger.error(f"Generation Error: {e}")
        return {"error": str(e)}

# تشغيل خادم RunPod Serverless
runpod.serverless.start({"handler": handler})
