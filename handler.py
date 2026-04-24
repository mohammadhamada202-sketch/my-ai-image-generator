import runpod
import torch
import base64
import os
from io import BytesIO
from openai import OpenAI  # إضافة مكتبة OpenAI
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# إعداد عميل OpenAI
# يفضل وضع المفتاح في Environment Variables في RunPod باسم OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"))

MODEL_CACHE_DIR = "/workspace/models"

# [span_1](start_span)مصفوفة الستايلات المحدثة لنتائج خرافية[span_1](end_span)
STYLE_MODIFIERS = {
    "realistic": "photorealistic, highly detailed skin texture, 8k uhd, masterpiece, raw photo, soft global illumination",
    "anime": "high-quality anime style, studio ghibli inspired, vibrant colors, clean lineart, sharp focus",
    "cinematic": "cinematic movie still, anamorphic lens flare, dramatic lighting, moody atmosphere, 35mm lens",
    "cartoon": "high-quality 2d animation style, vibrant, clean vector lines, expressive character design",
    "pixar": "disney pixar 3d style, ultra-detailed render, subsurface scattering, cute aesthetics, 4k"
}

def enhance_prompt_via_gpt(user_input):
    """وظيفة الترجمة والتحسين الذكي باستخدام GPT-4o-mini"""
    try:
        instruction = (
            "You are a professional Prompt Engineer. Translate the user input (any language/slang) "
            "to a detailed English prompt for Stable Diffusion XL. Add artistic keywords like "
            "lighting, camera angles, and textures to make it 'masterpiece' quality. "
            "Output ONLY the final English prompt."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT Error: {e}")
        return user_input # العودة للنص الأصلي في حال الخطأ

# [span_2](start_span)تحميل الموديل خارج الـ handler لتجنب الـ Cold Start[span_2](end_span)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

def handler(job):
    try:
        [span_3](start_span)job_input = job['input'][span_3](end_span)
        [span_4](start_span)user_prompt = job_input.get('prompt', '')[span_4](end_span)
        [span_5](start_span)style = job_input.get('style', 'realistic')[span_5](end_span)
        [span_6](start_span)width = job_input.get('width', 1024)[span_6](end_span)
        [span_7](start_span)height = job_input.get('height', 1024)[span_7](end_span)

        # 1. الترجمة والتحسين الذكي
        enhanced_prompt = enhance_prompt_via_gpt(user_prompt)
        
        # 2. دمج الستايل التقني
        [span_8](start_span)modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])[span_8](end_span)
        final_prompt = f"{enhanced_prompt}, {modifier}"

        # 3. [span_9](start_span)توليد الصورة[span_9](end_span)
        image = pipe(
            prompt=final_prompt,
            negative_prompt="low quality, blurry, bad anatomy, deformed faces, extra fingers, watermark, text",
            num_inference_steps=35,
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]

        # 4. [span_10](start_span)تحويل الصورة إلى Base64[span_10](end_span)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        [span_11](start_span)return {"image_base64": img_str}[span_11](end_span)

    except Exception as e:
        [span_12](start_span)return {"error": str(e)}[span_12](end_span)

# [span_13](start_span)تشغيل السيرفر[span_13](end_span)
runpod.serverless.start({"handler": handler})
