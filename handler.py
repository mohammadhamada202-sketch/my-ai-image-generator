import runpod
import torch
import huggingface_hub
from diffusers import StableDiffusionXLPipeline

# --- حل مشكلة ImportError: cannot import name 'cached_download' ---
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# --- دالة إعداد الموديل (تشتغل مرة واحدة عند إقلاع السيرفر) ---
def setup():
    model_id = "SG161222/RealVisXL_V4.0"
    print(f"--- [1/4] جاري تحميل الموديل: {model_id} ---")
    
    # تحميل الموديل مع إعدادات الذاكرة لكرت 4090
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    
    # تحسين الأداء (اختياري لكن مفيد جداً)
    pipe.enable_xformers_memory_efficient_attention()
    
    print("--- [2/4] تم التحميل بنجاح! السيرفر جاهز لاستلام الطلبات ---")
    return pipe

# تشغيل الإعداد
pipe = setup()

# --- دالة المعالجة (تشتغل عند كل طلب صورة جديد) ---
def handler(job):
    job_input = job["input"]
    
    # استخراج البرومبت (ووضع قيمة افتراضية إذا كان فارغاً)
    prompt = job_input.get("prompt", "A high-tech futuristic city")
    negative_prompt = job_input.get("negative_prompt", "(low quality, worst quality:1.2), blurry, distorted")
    
    print(f"--- [3/4] جاري العمل على طلبك: {prompt} ---")
    
    # توليد الصورة
    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # حفظ الصورة مؤقتاً في السيرفر
        output_path = "/tmp/output.png"
        image.save(output_path)
        
        print("--- [4/4] اكتملت الصورة بنجاح! ---")
        
        # إرجاع النتيجة (يمكنك تعديل هذا الجزء لرفع الصورة لـ S3 أو إرسالها كـ Base64)
        return {"status": "success", "message": "Image generated successfully", "image_url": output_path}
    
    except Exception as e:
        print(f"❌ خطأ أثناء التوليد: {str(e)}")
        return {"status": "error", "message": str(e)}

# بدء تشغيل خدمة RunPod Serverless
runpod.serverless.start({"handler": handler})
