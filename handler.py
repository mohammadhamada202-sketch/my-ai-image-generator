import runpod
import torch
import huggingface_hub
import base64
import os
from diffusers import StableDiffusionXLPipeline

# --- [1] حل مشكلة الإصدارات في HuggingFace ---
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# --- [2] دالة إعداد الموديل عند الإقلاع ---
def setup():
    model_id = "SG161222/RealVisXL_V4.0"
    print(f"--- [START] جاري تحميل الموديل: {model_id} ---")
    
    # تحميل الموديل بإعدادات الذاكرة لكرت 4090
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    
    # محاولة تفعيل xformers لتسريع الأداء
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("--- [OK] تم تفعيل xformers بنجاح ---")
    except Exception as e:
        print(f"--- [INFO] سيتم العمل بدون xformers ---")
    
    print("--- [READY] السيرفر جاهز لاستلام الطلبات ---")
    return pipe

# تشغيل الإعداد مرة واحدة
pipe = setup()

# --- [3] دالة المعالجة لكل طلب جديد ---
def handler(job):
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "A realistic photo of Burj Khalifa")
    negative_prompt = job_input.get("negative_prompt", "(low quality, worst quality:1.2), blurry, distorted")
    
    print(f"--- [LOG] جاري رسم: {prompt} ---")
    
    try:
        # توليد الصورة
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # حفظ الصورة مؤقتاً في السيرفر
        temp_path = "/tmp/output.png"
        image.save(temp_path)
        
        # تحويل الصورة إلى Base64
        with open(temp_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # حذف الصورة المؤقتة لتوفير المساحة
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        print("--- [DONE] تم توليد الصورة وتحويلها لـ Base64 بنجاح! ---")
        
        # إرجاع النتيجة للمتصفح
        return {
            "status": "success",
            "image_base64": encoded_string
        }
    
    except Exception as e:
        print(f"❌ خطأ تقني: {str(e)}")
        return {"status": "error", "message": str(e)}

# ربط الكود بـ RunPod
runpod.serverless.start({"handler": handler})
