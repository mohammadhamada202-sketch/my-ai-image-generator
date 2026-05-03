import runpod
import torch
import base64
import gc
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# استيراد الملفات المساعدة
from styles_config import STYLE_PROMPTS, NEGATIVE_PROMPT
from dimensions_helper import get_dimensions
from translator_helper import translate_and_optimize

# إعداد النماذج (تحميل مرة واحدة عند بدء التشغيل)
MODEL_ID = "SG161222/RealVisXL_V4.0"
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe).to("cuda")

def handler(job):
    try:
        # تنظيف الذاكرة قبل كل عملية
        torch.cuda.empty_cache()
        gc.collect()

        job_input = job['input']
        mode = job_input.get('mode', 'text')
        user_prompt = job_input.get('prompt', '')
        style = job_input.get('style', 'realistic')
        
        # 1. تنفيذ الترجمة والتحسين (الاستدعاء السحري)
        optimized_prompt = translate_and_optimize(user_prompt)
        
        # 2. جلب الستايل المناسب
        style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS['realistic'])

        # 3. توليد الصورة بناءً على المود (نص أو صورة لصورة)
        if mode == 'text':
            width, height = get_dimensions(job_input)
            # دالة التوليد (تأكد أن ملف text_generator سليم)
            from text_generator import generate_from_text
            output_img = generate_from_text(pipe, optimized_prompt, style_prompt, NEGATIVE_PROMPT, width, height)
        else:
            image_b64 = job_input.get('image')
            from avatar_generator import generate_avatar
            output_img = generate_avatar(img_pipe, image_b64, optimized_prompt, style_prompt, NEGATIVE_PROMPT)

        # 4. تحويل الصورة النهائية لـ Base64 لإرسالها للموقع
        buffered = BytesIO()
        output_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # العودة بالصورة مباشرة (هذا يضمن ظهورها في الـ Frontend الخاص بك)
        return img_str

    except Exception as e:
        print(f"Handler Error: {str(e)}")
        return {"error": str(e)}

# بدء السيرفر
runpod.serverless.start({"handler": handler})
