import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import runpod
import torch
import io
import base64
from diffusers import StableDiffusionXLPipeline

# تحميل الموديل مرة واحدة فقط عند الإقلاع
def load_model():
    print("--- [1/4] جاري تشغيل المحرك وتحميل RealVisXL... ---")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to("cuda")
        print("--- [2/4] السيرفر جاهز تماماً للرسم ---")
        return pipe
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {str(e)}")
        return None

pipe = load_model()

def handler(job):
    try:
        prompt = job['input'].get('prompt', 'Professional realistic photo')
        print(f"--- [3/4] جاري إنشاء صورة لـ: {prompt} ---")
        
        # تنفيذ الرسم بـ 25 خطوة لدقة عالية
        with torch.inference_mode():
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        
        # تحويل الصورة إلى Base64
        buffer = io.BytesIO()
        image.save(buffer, format="WebP", quality=90)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print("--- [4/4] تم الإنشاء بنجاح وإرسال الصورة للموقع ---")
        return {"image": image_base64}
    except Exception as e:
        print(f"❌ خطأ أثناء التنفيذ: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
