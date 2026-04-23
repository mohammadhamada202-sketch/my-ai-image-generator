import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import runpod
import torch
import io
import base64
from diffusers import StableDiffusionXLPipeline

# متغير عالمي للموديل
pipe = None

def load_model():
    global pipe
    print("--- [1/4] جاري تحميل الموديل في الذاكرة... ---")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to("cuda")
        print("--- [2/4] الموديل جاهز تماماً! ---")
    except Exception as e:
        print(f"❌ فشل تحميل الموديل: {str(e)}")

# محاولة التحميل عند التشغيل
load_model()

def handler(job):
    global pipe
    try:
        # إذا كان الموديل لم يتحمل بعد، نحاول تحميله مرة أخرى
        if pipe is None:
            load_model()
            if pipe is None:
                return {"error": "الموديل لا يزال قيد التحميل، جرب بعد ثوانٍ"}

        prompt = job['input'].get('prompt', 'Professional photo')
        print(f"--- [3/4] جاري رسم: {prompt} ---")
        
        with torch.inference_mode():
            # تأكدنا هنا أن pipe ليس None قبل استدعائه
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="WebP", quality=90)
        
        print("--- [4/4] تم بنجاح! ---")
        return {"image": base64.b64encode(buffer.getvalue()).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

