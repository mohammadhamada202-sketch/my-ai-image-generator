import os
# تعطيل التنبيهات والبحث عن كروت شاشة غير متوافقة قبل استدعاء المكتبات
os.environ["DIFFUSERS_NO_ADVICE_LOGS"] = "1"
os.environ["ACCELERATE_USE_CPU"] = "False"

import runpod
import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

# 1. إعداد الجهاز والموديل
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SG161222/RealVisXL_V4.0"

print("--- جاري تجهيز الذكاء الاصطناعي... يرجى الانتظار ---")

try:
    # تحميل الموديل بإعدادات الذاكرة المحسنة
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    # استخدام محرك رسم سريع ودقيق
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # إعداد المترجم
    translator = GoogleTranslator(source='auto', target='en')
    print("✅ السيرفر جاهز الآن لاستقبال طلباتك")
except Exception as e:
    print(f"❌ خطأ أثناء تحميل الموديل: {e}")

# 2. قاموس الستايلات لتعزيز جودة الصور
STYLES = {
    "realistic": "photorealistic, 8k uhd, cinematic lighting, highly detailed, masterwork",
    "anime": "anime art style, vibrant, high contrast, clean lines, studio quality",
    "digital_art": "digital painting, artstation style, smooth gradients, sharp focus",
    "3d_render": "octane render, unreal engine 5, volumetric lighting, hyper-realistic 3d"
}

def handler(job):
    try:
        job_input = job['input']
        
        # استلام البيانات من الموقع
        prompt = job_input.get('prompt', 'A beautiful landscape')
        style = job_input.get('style', 'realistic')
        orientation = job_input.get('orientation', 'square')
        quality = job_input.get('quality', 'HD')

        # ترجمة الوصف للعربية تلقائياً
        en_prompt = translator.translate(prompt)
        
        # دمج الستايل المختار
        style_prompt = STYLES.get(style, STYLES["realistic"])
        final_prompt = f"{en_prompt}, {style_prompt}"

        # ضبط المقاسات (SDXL Optimized)
        dims = {
            "portrait": (832, 1216), 
            "landscape": (1216, 832), 
            "square": (1024, 1024)
        }
        width, height = dims.get(orientation, (1024, 1024))

        # ضبط الجودة (عدد الخطوات)
        num_steps = 40 if quality == "4K" else 25

        # عملية توليد الصورة
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                negative_prompt="deformed, blurry, low quality, bad anatomy, text, watermark",
                num_inference_steps=num_steps,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]
        
        # تحويل الصورة إلى Base64 بجودة عالية
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=95)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {"status": "success", "image": img_b64}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# تشغيل السيرفر
runpod.serverless.start({"handler": handler})
