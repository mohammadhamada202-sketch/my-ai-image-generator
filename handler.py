import os
import sys

# 1. رقعة إصلاح حاسمة لـ huggingface_hub قبل أي استدعاء آخر
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except:
    pass

# 2. إعدادات البيئة لمنع التعارضات
os.environ["DIFFUSERS_NO_ADVICE_LOGS"] = "1"
os.environ["ACCELERATE_USE_CPU"] = "False"

import runpod
import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

# 3. تجهيز الموديل والجهاز
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SG161222/RealVisXL_V4.0"

print("--- جاري بدء تشغيل السيرفر وتحميل الموديل ---")

try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    # تحسين المحرك لصور واقعية وسريعة
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # إعداد المترجم
    translator = GoogleTranslator(source='auto', target='en')
    print("✅ تم بنجاح: السيرفر جاهز لاستقبال الطلبات")
except Exception as e:
    print(f"❌ خطأ في تحميل الموديل: {str(e)}")

# 4. الستايلات المدعومة
STYLES = {
    "realistic": "photorealistic, 8k uhd, cinematic lighting, raw photo, highly detailed",
    "anime": "anime style, vibrant colors, high resolution, detailed line art",
    "digital_art": "digital painting, artstation, concept art, sharp focus",
    "3d_render": "octane render, unreal engine 5, 8k, volumetric lighting"
}

def handler(job):
    try:
        job_input = job['input']
        
        # قراءة المدخلات من الموقع
        prompt = job_input.get('prompt', 'A beautiful landscape')
        style = job_input.get('style', 'realistic')
        orientation = job_input.get('orientation', 'square')
        quality = job_input.get('quality', 'HD')

        # الترجمة التلقائية
        en_prompt = translator.translate(prompt)
        
        # دمج الستايل
        style_suffix = STYLES.get(style, STYLES["realistic"])
        final_prompt = f"{en_prompt}, {style_suffix}"

        # إعدادات الأبعاد (SDXL Optimized)
        dims = {
            "portrait": (832, 1216), 
            "landscape": (1216, 832), 
            "square": (1024, 1024)
        }
        width, height = dims.get(orientation, (1024, 1024))

        # جودة 4K تزيد من دقة التفاصيل (Steps)
        num_steps = 40 if quality == "4K" else 25

        # عملية الرسم
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                negative_prompt="canvas, plastic, blurry, low quality, deformed, text, watermark",
                num_inference_steps=num_steps,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]
        
        # تحويل الصورة إلى Base64
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=95)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {"status": "success", "image": img_str}

    except Exception as e:
        return {"status": "error", "message": str(e)}
        

# تشغيل محرك RunPod
runpod.serverless.start({"handler": handler})
