import runpod
import torch
import base64
import gc
import os
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# --- [1. إعدادات المسارات والتخزين الدائم] ---
CACHE_DIR = "/workspace/models"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- [2. الستايلات مع ميزة الـ Enhance الاحترافي] ---
STYLE_ENHANCERS = {
    "photorealistic": "high-end cinematic portrait, extremely detailed facial features, 8k resolution, shot on 85mm lens, f/1.8, stunning soft bokeh, studio professional lighting, masterpiece, hyper-realistic skin texture",
    "anime": "high-quality anime illustration, vibrant studio Ghibli colors, clean cinematic lineart, masterpiece, 4k, expressive lighting, official art style",
    "3d_render": "octane render, unreal engine 5 style, highly detailed 3D character, ray tracing, cinematic rim lighting, polished textures, volumetric fog",
    "sketch": "professional charcoal graphite drawing, artistic hatching, elegant pencil strokes, high contrast, textured paper aesthetic",
    "pixel_art": "meticulous 8-bit pixel art, high-quality retro game aesthetic, sharp vibrant pixels, iconic character design"
}

AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, text, error, cropped, blurry, grainy, ugly, deformed, duplicate, watermark"

# الموديلات الرسمية والمجربة
MODELS_CONFIG = {
    "realism": "SG161222/RealVisXL_V4.0",
    "anime": "cagliostrolab/animagine-xl-3.1"
}
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# تعريف المتغيرات العالمية للمحرك
active_model_id = None
global_pipe = None
global_img_pipe = None

# --- [3. نظام المترجم والمحسن] ---
try:
    from translator_helper import translate_and_optimize
    print("--- [SYSTEM] Translator helper loaded successfully ---")
except ImportError:
    print("--- [WARNING] translator_helper.py not found. Using fallback ---")
    def translate_and_optimize(p): return p

def get_engine(style):
    global active_model_id, global_pipe, global_img_pipe
    
    # اختيار الموديل (الأنمي للأنمي والكرتون، والواقعي للبقية)
    target_id = MODELS_CONFIG["anime"] if style in ['anime', 'cartoon'] else MODELS_CONFIG["realism"]
    
    if active_model_id == target_id and global_pipe is not None:
        return global_pipe, global_img_pipe

    print(f"--- Switching/Loading Engine to: {target_id} ---")
    
    # تنظيف الذاكرة قبل التبديل
    if global_pipe is not None:
        del global_pipe
        del global_img_pipe
        torch.cuda.empty_cache()
        gc.collect()

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    
    # تحميل الموديل (تم إزالة variant="fp16" لضمان التوافق مع Animagine)
    global_pipe = StableDiffusionXLPipeline.from_pretrained(
        target_id, 
        vae=vae, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        cache_dir=CACHE_DIR
    ).to("cuda")
    
    global_pipe.enable_xformers_memory_efficient_attention()
    global_img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(global_pipe).to("cuda")
    
    active_model_id = target_id
    return global_pipe, global_img_pipe

def handler(job):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        style = job_input.get('style', 'photorealistic')
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')

        # 1. الترجمة والتحسين (تحويل من العربي إلى إنجليزي محسن)
        translated_prompt = translate_and_optimize(user_prompt)

        # 2. الـ Enhance الاحترافي (حقن المواصفات التقنية في البرومبت)
        enhancer = STYLE_ENHANCERS.get(style, STYLE_ENHANCERS['photorealistic'])
        final_optimized_prompt = f"{enhancer}, {translated_prompt}, highly detailed, intricate details"
        
        print(f"--- [ENHANCED PROMPT]: {final_optimized_prompt} ---")

        # 3. جلب المحرك المناسب (الواقعي أو الأنمي)
        current_pipe, current_img_pipe = get_engine(style)
        
        # 4. التوليد
        if mode == 'text':
            output_img = current_pipe(
                prompt=final_optimized_prompt, 
                negative_prompt=AVATAR_NEGATIVE_PROMPT,
                num_inference_steps=40, # دقة عالية
                guidance_scale=12.0     # التزام قوي بالوصف
            ).images[0]
        else:
            image_b64 = job_input.get('image')
            # استدعاء مولد الأفاتار للصور
            from avatar_generator import generate_avatar
            output_img = generate_avatar(current_img_pipe, image_b64, final_optimized_prompt, style, AVATAR_NEGATIVE_PROMPT)

        # 5. تصدير النتيجة بصيغة PNG جودة 95%
        buffered = BytesIO()
        output_img.save(buffered, format="PNG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Runtime Error: {str(e)}")
        return {"error": str(e)}

# بدء تشغيل السيرفر
runpod.serverless.start({"handler": handler})
