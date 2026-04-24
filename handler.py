import runpod
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from googletrans import Translator
import base64
from io import BytesIO

# إعداد المترجم
translator = Translator()

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        aspect_ratio = job_input.get('aspect_ratio', 'square')
        quality = job_input.get('quality', 'HD')

        print(f"--- [START] جاري معالجة الطلب: {prompt} ---")

        # 1. الترجمة التلقائية
        try:
            detected = translator.detect(prompt)
            if detected.lang != 'en':
                prompt = translator.translate(prompt, dest='en').text
                print(f"--- تم الترجمة إلى: {prompt} ---")
        except Exception as e:
            print(f"--- فشلت الترجمة، سيتم استخدام النص الأصلي: {str(e)} ---")

        # 2. إضافة لمسة الستايل للوصف
        style_prompts = {
            "realistic": "extremely detailed, 8k uhd, realistic, masterpiece, professional photography",
            "anime": "anime style, vibrant colors, high resolution, detailed eyes",
            "cartoon": "cartoon style, 3d render, cute, bright colors",
            "pixar": "disney pixar style, 3d animation, highly detailed, cute characters"
        }
        full_prompt = f"{prompt}, {style_prompts.get(style, '')}"

        # 3. تحميل الموديل (مع معالجة الأخطاء)
        print("--- جاري تحميل الموديل من HuggingFace... ---")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # تفعيل xformers لتسريع العمل وتقليل استهلاك الذاكرة
        pipe.enable_xformers_memory_efficient_attention()

        # 4. تحديد المقاسات
        dims = {"square": (1024, 1024), "landscape": (1216, 832), "portrait": (832, 1216)}
        width, height = dims.get(aspect_ratio, (1024, 1024))
        
        # 5. تحديد عدد الخطوات حسب الجودة
        steps = 50 if quality == "4K" else 30

        # 6. توليد الصورة
        print(f"--- جاري الرسم الآن (Steps: {steps})... ---")
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=7.5
        ).images[0]

        # 7. تحويل الصورة إلى Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("--- [DONE] تم توليد الصورة بنجاح! ---")
        return {"image_base64": img_str}

    except Exception as e:
        print(f"--- [CRITICAL ERROR]: {str(e)} ---")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
