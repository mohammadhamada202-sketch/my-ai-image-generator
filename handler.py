import os
import torch
import runpod
import base64
import logging
import io
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد السجلات لمراقبة أداء السيرفر في رومانيا
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- سحب المفتاح من إعدادات RunPod ---
# هذا السطر يبحث عن متغير اسمه OPENAI_API_KEY في إعدادات الـ Endpoint
api_key = os.getenv("OPENAI_API_KEY")

# التحقق من وجود المفتاح لمنع انهيار السيرفر عند الإقلاع
if not api_key:
    logger.error("--- [CRITICAL] OPENAI_API_KEY IS MISSING IN RUNPOD SETTINGS ---")
    client = None
else:
    # تهيئة عميل OpenAI
    client = OpenAI(api_key=api_key)

# تحميل موديل الرسم SDXL
logger.info("Starting Fresh SDXL Pipeline in Romania...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# تحسين سرعة الرسم باستخدام Scheduler متطور
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def handler(job):
    # التحقق من أن OpenAI جاهز للعمل
    if client is None:
        logger.error("Job failed: OpenAI client not initialized.")
        return {"error": "API Key is missing. Add OPENAI_API_KEY to RunPod Environment Variables."}

    job_input = job['input']
    user_prompt = job_input.get('prompt', '')
    
    logger.info(f"--- Processing Job: {user_prompt} ---")
    
    try:
        # إرسال الطلب لـ OpenAI للترجمة والتحسين
        # استخدمنا gpt-4o-mini لتقليل استهلاك الرصيد (الـ 5 دولار) وسرعة الاستجابة
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English and enhance as a professional SDXL prompt. Return only the English text."},
                {"role": "user", "content": user_prompt}
            ],
            timeout=15
        )
        
        final_prompt = response.choices[0].message.content.strip()
        logger.info(f"--- OpenAI Enhancement Success: {final_prompt} ---")
        
    except Exception as e:
        # في حال حدوث أي خطأ في الاتصال، نستخدم النص الأصلي لضمان عدم توقف الخدمة
        logger.error(f"--- OpenAI Connection Failed: {str(e)} ---")
        final_prompt = user_prompt

    # بدء عملية التوليد الصوري
    with torch.inference_mode():
        image = pipe(
            prompt=final_prompt,
            num_inference_steps=30,
            width=job_input.get('width', 1024),
            height=job_input.get('height', 1024)
        ).images[0]

    # تحويل الصورة الناتجة إلى صيغة Base64 لإرسالها إلى واجهة الموقع
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    logger.info("--- Job Completed Successfully ---")
    return {"image_base64": img_str}

# بدء تشغيل عامل RunPod (Serverless Worker)
runpod.serverless.start({"handler": handler})
