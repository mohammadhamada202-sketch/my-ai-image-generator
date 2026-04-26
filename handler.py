import os
import torch
import runpod
import base64
import logging
import io
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# 1. إعدادات النظام والسجلات (ضرورية لمراقبة المشكلة)
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. إعداد OpenAI (تأكد من وضع المفتاح الصحيح هنا)
# تأكد أن الحساب فيه رصيد (Credits) كافٍ
OPENAI_CLIENT = OpenAI(api_key="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")

# 3. تحميل الموديل (يتم مرة واحدة عند الإقلاع)
logger.info("Loading SDXL Pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def translate_via_openai(user_text):
    """ هذه الدالة المسؤولة عن التواصل مع OpenAI """
    try:
        logger.info(f"--- ATTEMPTING OPENAI CONNECTION FOR: {user_text} ---")
        
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Translate the user input to English and expand it into a detailed visual prompt for SDXL. Return ONLY the English text."
                },
                {"role": "user", "content": user_text}
            ],
            timeout=15 # وقت انتظار لضمان عدم تعليق السيرفر
        )
        
        translated_text = response.choices[0].message.content.strip()
        logger.info(f"--- OPENAI SUCCESS: {translated_text} ---")
        return translated_text
        
    except Exception as e:
        logger.error(f"--- OPENAI FAILED: {str(e)} ---")
        return user_text # في حال الفشل نستخدم النص الأصلي كأمان

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        
        # استدعاء دالة الترجمة (هنا يتم التواصل مع OpenAI)
        final_prompt = translate_via_openai(prompt)
        
        # عملية الرسم باستخدام النص المترجم
        logger.info("--- STARTING SDXL RENDERING ---")
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                num_inference_steps=30,
                width=job_input.get('width', 1024),
                height=job_input.get('height', 1024)
            ).images[0]

        # تحويل الصورة إلى Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        logger.error(f"--- HANDLER ERROR: {str(e)} ---")
        return {"error": str(e)}

# تشغيل خادم RunPod
runpod.serverless.start({"handler": handler})
