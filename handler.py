# ============================================================

# styles.py

# ============================================================

STYLE_MODIFIERS = {
“realistic”: (
“photorealistic, 8k uhd, raw photo, ultra-detailed, “
“highly professional, masterpiece, sharp focus, natural lighting”
),
“anime”: (
“anime style, studio ghibli, vibrant colors, “
“detailed lineart, high resolution, cel shading”
),
“cinematic”: (
“cinematic lighting, dramatic shadows, movie still, “
“35mm lens, sharp focus, anamorphic lens, film grain”
),
“cartoon”: (
“cartoon style, 2d animation, clean lines, “
“bold colors, playful design, flat shading”
),
“pixar”: (
“pixar animation style, 3d render, disney style, “
“cute character, subsurface scattering, 4k, volumetric lighting”
),
}

NEGATIVE_PROMPT = (
“low quality, blurry, distorted, low resolution, “
“bad hands, deformed faces, extra fingers, watermark, “
“text, signature, cropped, out of frame, worst quality”
)

# ============================================================

# model_loader.py

# ============================================================

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

MODEL_ID = “SG161222/RealVisXL_V4.0”
MODEL_CACHE_DIR = “/workspace/models”

def load_pipeline() -> StableDiffusionXLPipeline:
pipe = StableDiffusionXLPipeline.from_pretrained(
MODEL_ID,
torch_dtype=torch.float16,
variant=“fp16”,
cache_dir=MODEL_CACHE_DIR,
use_safetensors=True,
).to(“cuda”)

```
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++",
)

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

pipe.enable_attention_slicing()

return pipe
```

# ============================================================

# prompt_engine.py

# ============================================================

import os
from openai import OpenAI

def get_openai_client() -> OpenAI:
api_key = os.environ.get(“OPENAI_API_KEY”)
if not api_key:
raise ValueError(“OPENAI_API_KEY غير موجود في متغيرات البيئة”)
return OpenAI(api_key=api_key)

def is_english(text: str) -> bool:
ascii_count = sum(1 for c in text if ord(c) < 128)
return (ascii_count / max(len(text), 1)) > 0.85

def translate_to_english(client: OpenAI, text: str) -> str:
response = client.chat.completions.create(
model=“gpt-4o”,
messages=[
{
“role”: “system”,
“content”: (
“You are a professional translator. “
“Translate the user’s text to English. “
“Return ONLY the translated text, nothing else.”
),
},
{“role”: “user”, “content”: text},
],
temperature=0.3,
max_tokens=500,
)
return response.choices[0].message.content.strip()

def enhance_prompt(client: OpenAI, prompt: str, style: str) -> str:
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
),
},
{
“role”: “user”,
“content”: f”Enhance this image prompt: {prompt}”,
},
],
temperature=0.7,
max_tokens=300,
)
return response.choices[0].message.content.strip()

def prepare_prompt(raw_prompt: str, style: str) -> dict:
client = get_openai_client()

```
if is_english(raw_prompt):
    translated = raw_prompt
else:
    translated = translate_to_english(client, raw_prompt)

enhanced = enhance_prompt(client, translated, style)

return {
    "original": raw_prompt,
    "translated": translated,
    "enhanced": enhanced,
}
```

# ============================================================

# image_utils.py

# ============================================================

import base64
from io import BytesIO
from PIL import Image

def generate_image(
pipe,
enhanced_prompt: str,
style: str,
width: int = 1024,
height: int = 1024,
steps: int = 35,
guidance_scale: float = 7.5,
seed: int = None,
) -> Image.Image:
modifier = STYLE_MODIFIERS.get(style, STYLE_MODIFIERS[“realistic”])
full_prompt = f”{enhanced_prompt}, {modifier}”

```
generator = None
if seed is not None:
    generator = torch.Generator("cuda").manual_seed(seed)

result = pipe(
    prompt=full_prompt,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    width=width,
    height=height,
    generator=generator,
)

return result.images[0]
```

def image_to_base64(image: Image.Image, fmt: str = “PNG”) -> str:
buffered = BytesIO()
image.save(buffered, format=fmt)
return base64.b64encode(buffered.getvalue()).decode(“utf-8”)

# ============================================================

# handler.py

# ============================================================

import runpod

pipe = load_pipeline()

def handler(job: dict) -> dict:
try:
job_input = job.get(“input”, {})

```
    raw_prompt     = job_input.get("prompt", "")
    style          = job_input.get("style", "realistic")
    width          = int(job_input.get("width", 1024))
    height         = int(job_input.get("height", 1024))
    steps          = int(job_input.get("steps", 35))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    seed           = job_input.get("seed", None)

    if not raw_prompt:
        return {"error": "البرومبت فاضي، أرسل prompt في الـ input"}

    prompt_data     = prepare_prompt(raw_prompt, style)
    enhanced_prompt = prompt_data["enhanced"]

    image = generate_image(
        pipe=pipe,
        enhanced_prompt=enhanced_prompt,
        style=style,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    img_base64 = image_to_base64(image)

    return {
        "image_base64": img_base64,
        "prompt_info": {
            "original":   prompt_data["original"],
            "translated": prompt_data["translated"],
            "enhanced":   prompt_data["enhanced"],
        },
    }

except Exception as e:
    return {"error": str(e)}
```

runpod.serverless.start({“handler”: handler})
