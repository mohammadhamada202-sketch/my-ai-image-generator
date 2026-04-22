import runpod
import torch
import io
import base64
from diffusers import StableDiffusionXLPipeline
from deep_translator import GoogleTranslator

# رقعة الإصلاح الشهيرة
import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# تحميل الموديل
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
).to(device)

translator = GoogleTranslator(source='auto', target='en')

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', 'A beautiful landscape')
    try:
        en_prompt = translator.translate(prompt)
        with torch.inference_mode():
            image = pipe(prompt=en_prompt, num_inference_steps=25, width=768, height=768).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format='WebP', quality=90)
        return {"status": "success", "image": base64.b64encode(buf.getvalue()).decode('utf-8')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})