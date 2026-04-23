import runpod
import torch
import base64
import os
from googletrans import Translator
from diffusers import StableDiffusionXLPipeline

translator = Translator()

def setup():
    model_id = "SG161222/RealVisXL_V4.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    return pipe

pipe = setup()

def handler(job):
    job_input = job["input"]
    
    # 1. الترجمة التلقائية وتحسين البرومبت
    user_prompt = job_input.get("prompt", "")
    translated = translator.translate(user_prompt, dest='en').text
    
    # 2. إعداد الستايلات (Prompt Engineering)
    style = job_input.get("style", "realistic")
    styles_dict = {
        "realistic": ", photorealistic, 8k uhd, highly detailed, raw photo, master part",
        "anime": ", anime style, studio ghibli, high quality, vibrant colors",
        "cartoon": ", digital art, cartoon style, cute, bold lines",
        "pixar": ", 3d render, pixar style, disney, masterpiece, cgi, high detail"
    }
    
    final_prompt = translated + styles_dict.get(style, styles_dict["realistic"])
    
    # 3. إعداد المقاسات
    aspect_ratio = job_input.get("aspect_ratio", "square")
    dimensions = {
        "square": (1024, 1024),
        "landscape": (1216, 832),
        "portrait": (832, 1216)
    }
    width, height = dimensions.get(aspect_ratio, (1024, 1024))
    
    # 4. إعداد الجودة (Steps)
    quality = job_input.get("quality", "HD")
    steps = 50 if quality == "4K" else 30 # الـ 4K نزيد فيه خطوات المعالجة لدقة أعلى
    
    try:
        with torch.inference_mode():
            image = pipe(
                prompt=final_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=7.5
            ).images[0]
        
        temp_path = "/tmp/output.png"
        image.save(temp_path)
        
        with open(temp_path, "rb") as img_f:
            encoded = base64.b64encode(img_f.read()).decode('utf-8')
            
        return {"status": "success", "image_base64": encoded, "used_prompt": final_prompt}
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
