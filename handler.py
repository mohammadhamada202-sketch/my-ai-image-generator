import runpod
import torch
import base64
import gc
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# --- [1. إعدادات الستايلات المدمجة] ---
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait of the person, shot on 85mm lens, f/1.8, soft bokeh background, blurred backdrop, sharp focus on face, high-end studio lighting, 8k raw photo, extreme skin detail",
    "anime": "masterpiece, official anime style art of the person, high-quality 2D, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors",
    "3d_render": "highly detailed 3D Disney Pixar style avatar of the person, stylized digital character, Unreal Engine 5 render, cinematic lighting",
    "pixel_art": "genuine 8-bit pixel art avatar of the person, retro video game sprite, clean square pixels",
    "sketch": "raw charcoal sketch of the person on textured paper, hand-drawn artistic lines, rough hatching, high contrast",
    "abstract": "abstract digital art portrait of the person, geometric shapes, double exposure, vibrant neon color splashes"
}

AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry, grainy, deformed face, unrecognizable"

# --- [2. استيراد الملفات المساعدة مع نظام حماية] ---
try:
    from dimensions_helper import get_dimensions
    from translator_helper import translate_and_optimize
except ImportError:
    def get_dimensions(job_input): return (1024, 1024)
    def translate_and_optimize(prompt): return prompt

# --- [3. إعدادات الموديلات - تم استبدال الموديل المعطل بموديل رسمي مفتوح] ---
REALISM_MODEL = "SG161222/RealVisXL_V4.0" # بديل احترافي ومفتوح يتجاوز خطأ 401
ANIME_MODEL = "cagliostrolab/animagine-xl-3.1"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# متغيرات عالمية لإدارة الذاكرة
current_model_id = None
pipe = None
img_pipe = None

def load_model_on_demand(model_id):
    """تحميل الموديل المطلوب فقط ومسح القديم لتوفير الذاكرة"""
    global current_model_id, pipe, img_pipe
    
    if current_model_id == model_id:
        return pipe, img_pipe

    print(f"--- Switching Engine to: {model_id} ---")
    
    # تنظيف الذاكرة تماماً قبل تحميل الموديل الجديد
    if pipe is not None:
        del pipe
        del img_pipe
        torch.cuda.empty_cache()
        gc.collect()

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    
    img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to("cuda")
    current_model_id = model_id
    return pipe, img_pipe

def handler(job):
    try:
        # تنظيف سريع للذاكرة
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'photorealistic')
        
        # 1. اختيار الموديل المناسب بناءً على الستايل
        target_model = ANIME_MODEL if style in ['anime', 'cartoon'] else REALISM_MODEL
        active_pipe, active_img_pipe = load_model_on_demand(target_model)

        # 2. تحسين البرومبت والستايل
        optimized_prompt = translate_and_optimize(user_prompt)
        style_prompt = AVATAR_STYLES.get(style, AVATAR_STYLES['photorealistic'])

        # 3. التوليد
        if mode == 'text':
            width, height = get_dimensions(job_input)
            output_img = active_pipe(
                prompt=f"{style_prompt}, {optimized_prompt}",
                negative_prompt=AVATAR_NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_inference_steps=35
            ).images[0]
        else:
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            output_img = generate_avatar(active_img_pipe, image_b64, optimized_prompt, style, AVATAR_NEGATIVE_PROMPT)

        # 4. التصدير
        buffered = BytesIO()
        output_img.save(buffered, format="PNG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Handler Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
