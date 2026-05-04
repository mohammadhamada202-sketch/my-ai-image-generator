import runpod
import torch
import base64
import gc
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# استيراد الملفات المساعدة (الحفاظ على ميزاتك القديمة)
from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT
from dimensions_helper import get_dimensions
from translator_helper import translate_and_optimize

# 1. إعدادات الموديل العالمي (الدقة القصوى)
MODEL_ID = "Runware/Juggernaut-XL-v9" 
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

print(f"--- Loading Professional Model: {MODEL_ID} ---")

# تحميل الـ VAE والموديل بأعلى جودة (fp16) لضمان صفاء الصورة
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# إنشاء خط أنابيب الصورة-إلى-صورة (لأفاتار) من نفس الموديل لتوفير الذاكرة
img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to("cuda")

# تفعيل تقنيات تسريع الذاكرة لضمان استقرار السيرفر
pipe.enable_xformers_memory_efficient_attention()

def handler(job):
    try:
        # تنظيف الذاكرة (GPU) قبل كل عملية لتجنب الـ Crash
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text') # نص أو أفاتار
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'photorealistic')
        
        # 1. تنفيذ الترجمة والتحسين (ميزتك القديمة الرائعة)
        optimized_prompt = translate_and_optimize(user_prompt)
        
        # 2. جلب الستايل من ملف الأفاتار الجديد (الذي يحفظ الهوية)
        style_prompt = AVATAR_STYLES.get(style, AVATAR_STYLES['photorealistic'])

        # 3. التوليد بناءً على المود المختار
        if mode == 'text':
            # توليد صور من النص (مثل اللوحات أو المناظر الطبيعية)
            width, height = get_dimensions(job_input)[cite: 1]
            from text_generator import generate_from_text
            output_img = generate_from_text(pipe, optimized_prompt, style_prompt, AVATAR_NEGATIVE_PROMPT, width, height)
        else:
            # توليد الأفاتار (استخدام صورتك الشخصية)
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            # نمرر الـ style كـ key وليس كبرومبت ليختار الـ strength الصحيح
            output_img = generate_avatar(img_pipe, image_b64, optimized_prompt, style, AVATAR_NEGATIVE_PROMPT)

        # 4. تحويل الصورة النهائية لـ Base64 بأعلى جودة (Quality 95)
        buffered = BytesIO()
        output_img.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # العودة بالصورة مباشرة للـ Frontend
        return img_str

    except Exception as e:
        print(f"--- Global Handler Error: {str(e)} ---")
        return {"error": str(e)}

# بدء السيرفر في وضع Serverless
runpod.serverless.start({"handler": handler})
