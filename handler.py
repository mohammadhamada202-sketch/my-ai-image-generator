import os
import torch
import runpod
import base64
import logging
from io import BytesIO
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# 1. إعدادات النظام والسجلات
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. إعداد OpenAI (تأكد من وضع مفتاحك هنا)
client = OpenAI(api_key="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")

# 3. تحميل موديل الصور (SDXL) - يتم التحميل مرة واحدة عند تشغيل الحاوية
logger.info("Loading SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# استخدام محرك سرعة لتحسين الأداء
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# قاموس الستايلات
STYLE_MAP = {
    "realistic": "photorealistic, 8k, raw photo, masterpiece",
    "cinematic": "cinematic movie still, moody lighting, epic composition",
    "anime": "anime style art, vibrant colors, studio ghibli aesthetic",
    "cartoon": "3d cartoon disney style, vibrant, cute",
    "pixar": "high-end 3d render, pixar movie aesthetics, soft lighting"
}

def handler(job):
    """
    الدالة الرئيسية التي تستقبل الطلبات من الموقع
    """
    try:
        job_input = job['input']
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        width = job_input.get('width', 1024)
        height = job_input.get('height', 1024)

        # --- المرحلة الأولى: OpenAI (الترجمة والتحسين) ---
        # إرسال إشارة للموقع بأننا بدأنا التواصل مع OpenAI
        runpod.serverless.progress(job, "OPENAI_PROCESSING")
        logger.info(f"Connecting to OpenAI for prompt: {user_prompt}")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "Translate to English and expand into a professional SDXL visual prompt. Return only the English text."
                    },
                    {"role": "user", "content": user_prompt}
                ],
                timeout=15
            )
            final_prompt = response.choices[0].message.content.strip()
            logger.info(f"OpenAI Result: {final_prompt}")
        except Exception as e:
            logger.error(f"OpenAI Error: {e}")
            final_prompt = user_prompt # العودة للأصل في حال فشل OpenAI

        # --- المرحلة الثانية: محرك الصور (الرسم) ---
        # إرسال إشارة للموقع بأننا انتهينا من الترجمة وبدأنا الرسم الفعلي
        runpod.serverless.progress(job, "DRAWING_PROCESSING")
        
        full_prompt = f"{final_prompt}, {STYLE_MAP.get(style, '')}"
        logger.info(f"Rendering image with: {full_prompt}")

        with torch.inference_mode():
            image = pipe(
                prompt=full_prompt,
                num_inference_steps=30,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]

        # --- المرحلة الثالثة: تحويل الصورة وإرسالها ---
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": image_base64}

    except Exception as e:
        logger.error(f"Handler Error: {e}")
        return {"error": str(e)}

# تشغيل خادم RunPod
runpod.serverless.start({"handler": handler})
