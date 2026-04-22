import runpod
import torch
import io
import base64
import os
from PIL import Image

# منع الـ Crash الناتج عن تعريفات الـ XPU
os.environ["ACCELERATE_USE_CPU"] = "False"

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    translator = GoogleTranslator(source='auto', target='en')
    print("✅ تم تحميل الموديل والمترجم بنجاح")
except Exception as e:
    print(f"❌ خطأ كارثي أثناء التحميل: {str(e)}")

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', 'A beautiful landscape')
        style = job_input.get('style', 'realistic')
        orientation = job_input.get('orientation', 'square')
        
        # ترجمة
        en_prompt = translator.translate(prompt)
        
        # مقاسات
        dims = {"portrait": (832, 1216), "landscape": (1216, 832), "square": (1024, 1024)}
        width, height = dims.get(orientation, (1024, 1024))

        with torch.inference_mode():
            image = pipe(
                prompt=f"{en_prompt}, {style}, highly detailed, masterpiece",
                num_inference_steps=25,
                width=width,
                height=height
            ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=90)
        return {"status": "success", "image": base64.b64encode(buf.getvalue()).decode('utf-8')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
