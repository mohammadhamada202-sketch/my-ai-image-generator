import runpod
import torch
import huggingface_hub
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
    
    # --- [3] محاولة تفعيل xformers بأمان ---
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("--- [OK] تم تفعيل xformers بنجاح لتسريع الأداء ---")
    except Exception as e:
        print(f"--- [INFO] سيتم العمل بدون xformers (السبب: {str(e)}) ---")
    
    print("--- [READY] السيرفر جاهز تماماً لاستلام الطلبات ---")
    return pipe

# تشغيل الإعداد مرة واحدة فقط
pipe = setup()

# --- [4] دالة المعالجة لكل طلب جديد ---
def handler(job):
    job_input = job["input"]
    
    # جلب البرومبت أو استخدام قيم افتراضية
    prompt = job_input.get("prompt", "A high-tech futuristic city")
    negative_prompt = job_input.get("negative_prompt", "(low quality, worst quality:1.2), blurry, distorted")
    
    print(f"--- [LOG] جاري العمل على برومبت: {prompt} ---")
    
    try:
        # توليد الصورة
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # حفظ الصورة في المجلد المؤقت الخاص بـ RunPod
        output_path = "/tmp/output.png"
        image.save(output_path)
        
        print("--- [DONE] تم توليد الصورة بنجاح! ---")
        
        # إرجاع النتيجة
        return {
            "status": "success",
            "message": "Image generated",
            "image_path": output_path
        }
    
    except Exception as e:
        print(f"❌ خطأ تقني أثناء التوليد: {str(e)}")
        return {"status": "error", "message": str(e)}

# ربط الكود بـ RunPod
runpod.serverless.start({"handler": handler})
