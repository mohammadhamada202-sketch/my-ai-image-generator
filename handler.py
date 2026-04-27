import runpod
import torch
import base64
from io import BytesIO
import os
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from openai import OpenAI

# المسار الخاص بالـ Network Volume لتخزين الموديل

MODEL_CACHE_DIR = “/workspace/models”

# قاموس الستايلات الفنية المدعومة

STYLE_MODIFIERS = {
“realistic”: “photorealistic, 8k uhd, raw photo, ultra-detailed, highly professional, masterpiece”,
“anime”: “anime style, studio ghibli, vibrant colors, detailed lineart, high resolution”,
“cinematic”: “cinematic lighting, dramatic shadows, movie still, 35mm lens, sharp focus”,
“cartoon”: “cartoon style, 2d animation, clean lines, bold colors, playful design”,
“pixar”: “pixar animation style, 3d render, disney style, cute character, subsurface scattering, 4k”
}

NEGATIVE_PROMPT = (
“low quality, blurry, distorted, low resolution, “
“bad hands, deformed faces, extra fingers, watermark, “
“text, signature, cropped, out of frame, worst quality”
)

# ──────────────────────────────────────────

# تحميل الموديل مرة وحدة عند بدء السيرفر

# ──────────────────────────────────────────

pipe = StableDiffusionXLPipeline.from_pretrained(
“SG161222/RealVisXL_V4.0”,
torch_dtype=torch.float16,
variant=“fp16”,
cache_dir=MODEL_CACHE_DIR
).to(“cuda”)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

try:
pipe.enable_xformers_memory_efficient_attention()
except:
pass

# ──────────────────────────────────────────

# دوال الترجمة والـ Enhance

# ──────────────────────────────────────────

def is_english(text):
ascii_count = sum(1 for c in text if ord(c) < 128)
return (ascii_count / max(len(text), 1)) > 0.85

def translate_to_english(client, text):
response = client.chat.completions.create(
model=“gpt-4o”,
messages=[
{
“role”: “system”,
“content”: “You are a professional translator. Translate the user’s text to English. Return ONLY the translated text, nothing else.”
},
{“role”: “user”, “content”: text}
],
temperature=0.3,
max_tokens=500,
)
return response.choices[0].message.content.strip()

def enhance_prompt(client, prompt, style):
response = client.chat.completions.create(
model=“gpt-4o”,
messages=[
{
“role”: “system”,
“content”: (
“You are an expert AI image prompt engineer specializing in Stable Diffusion XL. “
“Your job is to take a simple image description and expand it into a highly detailed, “
“professional prompt that will generate stunning, high-quality images. “
f”The target style is: {style}. “
“Focus on: lighting, composition, mood, colors, textures, camera angle, and artistic quality. “
“Return ONLY the enhanced prompt text, no explanations, no bullet points.”
)
},
{“role”: “user”, “content”: f”Enhance this image prompt: {prompt}”}
],
temperature=0.7,
max_tokens=300,
)
return response.choices[0].message.content.strip()

def prepare_prompt(raw_prompt, style):
client = OpenAI(api_key=os.environ.get(“OPENAI_API_KEY”))

```
translated = raw_prompt if is_english(raw_prompt) else translate_to_english(client, raw_prompt)
enhanced   = enhance_prompt(client, translated, style)

return {"original": raw_prompt, "translated": translated, "enhanced": enhanced}
```

# ──────────────────────────────────────────

# الـ Handler الرئيسي

# ──────────────────────────────────────────

def handler(job):
try:
job_input = job[‘input’]

```
    user_prompt = job_input.get('prompt', '')
    style       = job_input.get('style', 'realistic')
    width       = job_input.get('width', 1024)
    height      = job_input.get('height', 1024)

    if not user_prompt:
        return {"error": "البرومبت فاضي"}

    # ترجمة + Enhance
    prompt_data     = prepare_prompt(user_prompt, style)
    enhanced_prompt = prompt_data["enhanced"]

    # إضافة الستايل modifier
    modifier    = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS["realistic"])
    full_prompt = f"{enhanced_prompt}, {modifier}"

    # توليد الصورة
    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=width,
        height=height
    ).images[0]

    # تحويل الصورة إلى Base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "image_base64": img_str,
        "prompt_info": {
            "original":   prompt_data["original"],
            "translated": prompt_data["translated"],
            "enhanced":   prompt_data["enhanced"],
        }
    }

except Exception as e:
    return {"error": str(e)}
```

# بدء تشغيل السيرفر

runpod.serverless.start({“handler”: handler})
