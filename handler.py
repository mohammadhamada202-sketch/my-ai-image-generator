import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# ==========================================
# 1. قسم الإعدادات (Styles Configuration)
# هذا القسم يمكنك تعديله مستقبلاً لإضافة أي ستايل جديد
# ==========================================
STYLES_CONFIG = {
    "realistic": {
        "positive": "hyper-realistic professional portrait, 8k, detailed skin, cinematic lighting",
        "negative": "cartoon, anime, low quality, distorted, drawing",
        "strength": 0.35
    },
    "anime": {
        "positive": "high-quality anime style, studio ghibli aesthetic, vibrant colors, clean lines",
        "negative": "photorealistic, 3d, messy, grainy, low quality",
        "strength": 0.60
    },
    "pixar": {
        "positive": "disney pixar 3d animation style, cute character, smooth textures, expressive eyes",
        "negative": "realistic, 2d, sketch, gritty, dark",
        "strength": 0.55
    },
    "cartoon": {
        "positive": "classic cartoon illustration, bold outlines, artistic simple colors, fun vibe",
        "negative": "photograph, 3d render, grainy, blurry, realistic anatomy",
        "strength": 0.70
    }
}

# ==========================================
# 2. محرك المعالجة (The Image Engine Class)
# هذا القسم هو "العضلات" التي تتعامل مع كرت الشاشة والموديل
# ==========================================
class AIImageEngine:
    def __init__(self, model_id="SG161222/RealVisXL_V4.0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache = "/workspace/models"
        
        # تحميل الموديل بنظام Img2Img لضمان أفضل تحويل للصور
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            cache_dir=self.model_cache
        ).to(self.device)
        
        # تحسين سرعة التوليد
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    def generate_transformed_image(self, base64_image, user_prompt, style_key):
        # جلب إعدادات الستايل المختار
        style = STYLES_CONFIG.get(style_key, STYLES_CONFIG["realistic"])
        
        # دمج برومبت المستخدم مع وصف الستايل الثابت
        full_prompt = f"{style['positive']}, {user_prompt}"
        
        # معالجة الصورة المرفوعة
        init_image = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB").resize((1024, 1024))
        
        # تنفيذ عملية التحويل البصري
        output = self.pipe(
            prompt=full_prompt,
            image=init_image,
            strength=style['strength'], # القوة المحددة لكل ستايل
            negative_prompt=style['negative'],
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # تحويل النتيجة لـ Base64 لإرسالها للموقع
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 3. المنسق الأساسي (The Handler)
# هذا القسم يستقبل الطلبات من RunPod ويوجهها للمحرك
# ==========================================

# تهيئة المحرك والذكاء الاصطناعي مرة واحدة عند بدء السيرفر (Warm Start)
engine = AIImageEngine()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_with_gpt(prompt):
    """استخدام GPT لترجمة وتحسين البرومبت قبل إرساله للرسم"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prompt engineer. Translate to English and enhance visual details. Return ONLY the enhanced text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except:
        return prompt # في حال فشل OpenAI نستخدم النص الأصلي

def handler(job):
    try:
        job_input = job['input']
        
        # جلب البيانات من الطلب القادم من الموقع
        image_data = job_input.get('image')
        raw_prompt = job_input.get('prompt', '')
        selected_style = job_input.get('style', 'realistic')

        if not image_data:
            return {"error": "No image provided for processing"}

        # 1. تحسين البرومبت (اختياري)
        enhanced_prompt = enhance_with_gpt(raw_prompt)

        # 2. معالجة الصورة عبر المحرك
        result_b64 = engine.generate_transformed_image(image_data, enhanced_prompt, selected_style)

        return {
            "image_base64": result_b64,
            "status": "success",
            "style_applied": selected_style
        }

    except Exception as e:
        return {"error": str(e)}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
