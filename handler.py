import runpod
import torch
import base64
import gc
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# استيراد الملفات المساعدة
from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT
from dimensions_helper import get_dimensions
from translator_helper import translate_and_optimize

# إعدادات الموديلات العالمية
REALISM_MODEL = "Runware/Juggernaut-XL-v9"  # للواقعية والـ 3D
ANIME_MODEL = "cagliostrolab/animagine-xl-3.1" # للأنمي والكرتون الاحترافي[cite: 1]
VAE_ID = "madebyollin/sdxl-vae-fp16-fix" # لإصلاح الألوان والدقة[cite: 1]

device = "cuda"

def load_pipeline(model_id):
    """دالة مساعدة لتحميل الموديلات بأفضل إعدادات للجودة"""
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention() # توفير الذاكرة[cite: 1]
    return pipe

# تحميل الموديلات عند بدء تشغيل السيرفر
print("--- Loading Realism Model ---")
pipe_realism = load_pipeline(REALISM_MODEL)
img_pipe_realism = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe_realism).to(device)

print("--- Loading Anime Model ---")
pipe_anime = load_pipeline(ANIME_MODEL)
img_pipe_anime = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe_anime).to(device)

def handler(job):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'photorealistic')
        
        # 1. تحسين البرومبت (الترجمة التلقائية)[cite: 1]
        optimized_prompt = translate_and_optimize(user_prompt)
        
        # 2. جلب الستايل المناسب
        style_prompt = AVATAR_STYLES.get(style, AVATAR_STYLES['photorealistic'])

        # 3. اختيار الموديل المناسب بناءً على الستايل[cite: 1]
        if style in ['anime', 'cartoon']:
            active_pipe = pipe_anime
            active_img_pipe = img_pipe_anime
            print(f"Using Anime Engine for style: {style}")
        else:
            active_pipe = pipe_realism
            active_img_pipe = img_pipe_realism
            print(f"Using Realism Engine for style: {style}")

        # 4. التوليد
        if mode == 'text':
            width, height = get_dimensions(job_input)
            from text_generator import generate_from_text
            output_img = generate_from_text(active_pipe, optimized_prompt, style_prompt, AVATAR_NEGATIVE_PROMPT, width, height)
        else:
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            # نمرر الموديل المختار للدالة
            output_img = generate_avatar(active_img_pipe, image_b64, optimized_prompt, style, AVATAR_NEGATIVE_PROMPT)

        # 5. التصدير بأعلى جودة
        buffered = BytesIO()
        output_img.save(buffered, format="PNG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Handler Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
