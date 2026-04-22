import runpod
import torch
import io
import base64
import os

# حل مشكلة التوافق مع تعريفات الكروت
os.environ["ACCELERATE_USE_CPU"] = "False"

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

# 1. تحميل الموديل مع إعدادات الجودة العالية
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SG161222/RealVisXL_V4.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# استخدام محرك (Scheduler) أسرع وأدق للصور الواقعية
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

translator = GoogleTranslator(source='auto', target='en')

# 2. قاموس الستايلات (Styles Dictionary)
STYLES = {
    "realistic": "extremely detailed, photorealistic, 8k uhd, cinematic lighting, masterpiece, highly intricate",
    "anime": "anime style, vibrant colors, detailed line art, studio ghibli style, high quality",
    "digital_art": "digital painting, concept art, sharp focus, vibrant, trending on artstation",
    "3d_render": "unreal engine 5 render, octane render, 8k, volumetric lighting, highly detailed 3d",
    "vintage": "vintage film style, grainy, 35mm lens, slight chromatic aberration, 1990s look"
}

def handler(job):
    job_input = job['input']
    
    # استلام المدخلات من الموقع
    user_prompt = job_input.get('prompt', 'A beautiful landscape')
    style = job_input.get('style', 'realistic')
    orientation = job_input.get('orientation', 'square') # square, portrait, landscape
    quality = job_input.get('quality', 'HD') # HD or 4K
    
    # ضبط المقاسات بناءً على اختيار اليوزر
    if orientation == "portrait":
        width, height = 832, 1216
    elif orientation == "landscape":
        width, height = 1216, 832
    else: # square
        width, height = 1024, 1024

    # ضبط عدد الخطوات بناءً على الجودة (4K يحتاج خطوات أكثر ودقة أعلى)
    steps = 40 if quality == "4K" else 25
    
    try:
        # ترجمة الوصف
        en_prompt = translator.translate(user_prompt)
        
        # دمج الوصف مع الستايل المختار
        style_suffix = STYLES.get(style, STYLES["realistic"])
        final_prompt = f"{en_prompt}, {style_suffix}"
        
        # توليد الصورة
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                negative_prompt="canvas, plastic, low quality, blurry, distorted, low resolution",
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=7.5
            ).images[0]
        
        # تحويل الصورة لـ Base64 بجودة عالية
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=95) # جودة 95 لتقليل الفقد
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {"status": "success", "image": image_base64}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
