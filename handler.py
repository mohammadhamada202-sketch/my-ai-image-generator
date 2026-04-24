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

        print(f"--- [START] معالجة طلب جديد: {prompt} ---")

        # 1. الترجمة التلقائية (من العربي للإنجليزي)
        try:
            detected = translator.detect(prompt)
            if detected.lang != 'en':
                prompt = translator.translate(prompt, dest='en').text
                print(f"--- الترجمة: {prompt} ---")
        except Exception as e:
            print(f"--- فشلت الترجمة، سيتم استخدام النص الأصلي: {str(e)} ---")

        # 2. إعدادات الستايل
        style_prompts = {
            "realistic": "extremely detailed, 8k uhd, realistic, masterpiece, professional photography",
            "anime": "anime style, vibrant colors, high resolution, detailed eyes",
            "cartoon": "cartoon style, 3d render, cute, bright colors",
            "pixar": "disney pixar style, 3d animation, highly detailed, cute characters"
        }
        full_prompt = f"{prompt}, {style_prompts.get(style, '')}"

        # 3. تحميل الموديل (من الملفات المحلية التي تم تجميدها)
        print("--- جاري تحميل الموديل من الذاكرة المحلية... ---")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            local_files_only=True, # لا تحمل من الإنترنت، الموديل موجود مسبقاً
            use_safetensors=True
        ).to("cuda")
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # تحسين الأداء
        if torch.cuda.is_available():
            pipe.enable_xformers_memory_efficient_attention()

        # 4. المقاسات والجودة
        dims = {"square": (1024, 1024), "landscape": (1216, 832), "portrait": (832, 1216)}
        width, height = dims.get(aspect_ratio, (1024, 1024))
        steps = 50 if quality == "4K" else 30

        # 5. توليد الصورة
        print(f"--- جاري الرسم الآن (Steps: {steps})... ---")
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=7.5
        ).images[0]

        # 6. التحويل لـ Base64 ليرسل للموقع
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("--- [DONE] تم توليد الصورة بنجاح! ---")
        return {"image_base64": img_str}

    except Exception as e:
        error_msg = f"--- [CRITICAL ERROR]: {str(e)} ---"
        print(error_msg)
        return {"error": str(e)}

# بدء عمل RunPod
runpod.serverless.start({"handler": handler})
