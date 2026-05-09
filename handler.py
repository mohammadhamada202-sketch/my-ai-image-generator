import runpod
import torch
import base64
import gc
import os
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# --- [1. إعداد مسار التخزين الدائم] ---
CACHE_DIR = "/workspace/models"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- [2. استيراد المترجم والمحسن - إلزامي] ---
try:
    from translator_helper import translate_and_optimize
    print("--- [SUCCESS] Translator System Imported ---")
except ImportError:
    print("--- [ERROR] translator_helper.py not found! Using fallback (no translation) ---")
    def translate_and_optimize(p): return p

# --- [3. الستايلات المدمجة] ---
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait of the person, shot on 85mm lens, f/1.8, soft bokeh background, blurred backdrop, sharp focus on face, high-end studio lighting, 8k raw photo, extreme skin detail",
    "anime": "masterpiece, official anime style art of the person, high-quality 2D, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors",
    "3d_render": "highly detailed 3D Disney Pixar style avatar of the person, stylized digital character, Unreal Engine 5 render, cinematic lighting",
    "sketch": "raw charcoal sketch of the person on textured paper, hand-drawn artistic lines, rough hatching, high contrast"
}
AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry, grainy, deformed face, unrecognizable"

# الموديلات الرسمية والمفتوحة
MODELS_CONFIG = {
    "realism": "SG161222/RealVisXL_V4.0",
    "anime": "cagliostrolab/animagine-xl-3.1"
}
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

active_model_id = None
pipe = None
img_pipe = None

def get_engine(style):
    global active_model_id, pipe, img_pipe
    target_id = MODELS_CONFIG["anime"] if style in ['anime', 'cartoon'] else MODELS_CONFIG["realism"]
    
    if active_model_id == target_id:
        return pipe, img_pipe

    print(f"--- Switching Engine to: {target_id} ---")
    if pipe is not None:
        del pipe
        del img_pipe
        torch.cuda.empty_cache()
        gc.collect()

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    new_pipe = StableDiffusionXLPipeline.from_pretrained(
        target_id, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir=CACHE_DIR
    ).to("cuda")
    
    new_pipe.enable_xformers_memory_efficient_attention()
    new_img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(new_pipe).to("cuda")
    
    active_model_id = target_id
    pipe, img_pipe = new_pipe, new_img_pipe
    return pipe, img_pipe

def handler(job):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        style = job_input.get('style', 'photorealistic')
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')

        # --- [إصلاح: استدعاء الترجمة والتحسين] ---
        # سيقوم بتحويل العربي لإنجليزي وإضافة كلمات تحسين الجودة
        optimized_prompt = translate_and_optimize(user_prompt)
        print(f"--- [TRANSLATION] From: '{user_prompt}' To: '{optimized_prompt}' ---")

        # تحميل المحرك
        current_pipe, current_img_pipe = get_engine(style)
        style_prompt = AVATAR_STYLES.get(style, AVATAR_STYLES['photorealistic'])
        
        if mode == 'text':
            output_img = current_pipe(
                prompt=f"{style_prompt}, {optimized_prompt}", 
                negative_prompt=AVATAR_NEGATIVE_PROMPT,
                num_inference_steps=35
            ).images[0]
        else:
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            # نمرر الـ Prompt المحسن للمولد
            output_img = generate_avatar(current_img_pipe, image_b64, optimized_prompt, style, AVATAR_NEGATIVE_PROMPT)

        buffered = BytesIO()
        output_img.save(buffered, format="PNG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Runtime Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
