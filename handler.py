import runpod
import torch
from diffusers import StableDiffusionXLPipeline

# تحسين الأداء وتحميل الموديل
def setup():
    model_id = "SG161222/RealVisXL_V4.0"
    print(f"--- [1/4] جاري تحميل الموديل: {model_id} ---")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    
    # تحسين استهلاك الذاكرة لكرت 4090
    pipe.enable_xformers_memory_efficient_attention()
    
    print("--- [2/4] تم التحميل بنجاح! السيرفر جاهز ---")
    return pipe

# استدعاء دالة التحميل مرة واحدة عند بدء السيرفر
pipe = setup()

def handler(job):
    # الحصول على البرومبت من الطلب
    job_input = job["input"]
    prompt = job_input.get("prompt", "A high-tech futuristic city")
    
    print(f"--- [3/4] جاري رسم: {prompt} ---")
    
    # عملية التوليد
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    # حفظ الصورة ورفعها (RunPod سيتكفل بتحويلها لرابط)
    image.save("/tmp/output.png")
    
    print("--- [4/4] اكتملت الصورة! ---")
    
    # يمكنك استخدام مكتبة لرفع الصورة أو إعادتها كـ base64
    # للتبسيط، سنفترض أن نظامك يستقبل الرد بنجاح
    return {"message": "Success", "image_path": "/tmp/output.png"}

runpod.serverless.start({"handler": handler})
