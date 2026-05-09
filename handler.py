import runpod
import torch
import base64
import gc
import os
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# --- [1. إعداد مسار التخزين الدائم] ---
# سيتم حفظ الموديلات هنا للأبد ولن تضطر لتحميلها مرة أخرى
CACHE_DIR = "/workspace/models"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- [2. الإعدادات والستايلات] ---
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait, shot on 85mm lens, f/1.8, soft bokeh background",
    "anime": "masterpiece, official anime style art, high-quality 2D, vibrant colors",
    "3d_render": "highly detailed 3D Disney Pixar style avatar",
    "sketch": "raw charcoal sketch, hand-drawn artistic lines"
}
AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, blurry, text, cropped"

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

    print(f"--- Loading Engine from Storage: {target_id} ---")
    
    if pipe is not None:
        del pipe
        del img_pipe
        torch.cuda.empty_cache()
        gc.collect()

    # استخدام cache_dir لضمان الحفظ في القرص الدائم
    vae = AutoencoderKL.from_pretrained(
        VAE_ID, 
        torch_dtype=torch.float16, 
        cache_dir=CACHE_DIR
    )
    
    new_pipe = StableDiffusionXLPipeline.from_pretrained(
        target_id, 
        vae=vae, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True,
        cache_dir=CACHE_DIR  # هنا يكمن السر: سيحفظ الموديل في ذاكرتك المشتراة
    ).to("cuda")
    
    new_pipe.enable_xformers_memory_efficient_attention()
    new_img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(new_pipe).to("cuda")
    
    active_model_id = target_id
    pipe, img_pipe = new_pipe, new_img_pipe
    return pipe, img_pipe

def handler(job):
    try:
        job_input = job['input']
        style = job_input.get('style', 'photorealistic')
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')

        current_pipe, current_img_pipe = get_engine(style)

        style_prompt = AVATAR_STYLES.get(style, AVATAR_STYLES['photorealistic'])
        
        if mode == 'text':
            output_img = current_pipe(
                prompt=f"{style_prompt}, {user_prompt}", 
                negative_prompt=AVATAR_NEGATIVE_PROMPT
            ).images[0]
        else:
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            output_img = generate_avatar(current_img_pipe, image_b64, user_prompt, style, AVATAR_NEGATIVE_PROMPT)

        buffered = BytesIO()
        output_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
