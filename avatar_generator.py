import torch
from PIL import Image, ImageOps, ImageFilter
import base64
from io import BytesIO
import gc

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt):
    try:
        # 1. فك تشفير الصورة الأصلية
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        init_image = ImageOps.exif_transpose(init_image)

        # 2. [فكرتك الذكية]: عمل زوم داخلي مع الحفاظ على التناسب
        # سنقوم بتوسيع الصورة قليلاً (Padding) ثم قصها بشكل مربع 1024x1024 
        # لضمان أن الوجه في المنتصف مع أخذ تفاصيل كافية من الخلفية
        width, height = init_image.size
        target_size = 1024
        
        # استخدام ImageOps.fit بذكاء (centering 0.5, 0.4 يركز على العيون والوجه عادةً)
        # لكن بـ LANCZOS عالي الجودة لضمان عدم ضياع التفاصيل
        processed_init = ImageOps.fit(init_image, (target_size, target_size), method=Image.LANCZOS, centering=(0.5, 0.4))

        # 3. جلب الإعدادات من ملف الستايلات
        from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT
        style_prompt = STYLE_ENHANCERS.get(style_key, STYLE_ENHANCERS["photorealistic"])
        
        # 4. هندسة البرومبت لتركيز الدقة على ملامح الوجه
        # أضفت كلمات لتعزيز "تفاصيل البشرة" و "العيون" بناءً على اقتراحك
        face_details = "extremely detailed eyes and skin, high resolution facial features, sharp focus on face"
        final_prompt = f"{prompt}, {style_prompt}, {face_details}, masterpiece"

        # 5. إعدادات القوة (Strength)
        # 0.55 هي النقطة المثالية: تسمح بتغيير الستايل دون فقدان الملامح البشرية
        custom_strength = 0.55 if style_key != "photorealistic" else 0.40
        
        # 6. عملية التوليد (High Fidelity)
        torch.cuda.empty_cache()
        output_image = img_pipe(
            prompt=final_prompt,
            negative_prompt=AVATAR_NEGATIVE_PROMPT,
            image=processed_init,
            strength=custom_strength,
            num_inference_steps=45,     # خطوات أكثر لتفاصيل أدق
            guidance_scale=10.0,        # توازن بين البرومبت والصورة
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]

        # 7. تنظيف الذاكرة
        gc.collect()
        
        return output_image

    except Exception as e:
        print(f"Avatar Generation Error: {str(e)}")
        return init_image # العودة للأصل في حال الخطأ
