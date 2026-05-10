import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO
import gc

# استدعاء من ملف ستايلات الأفاتار الخاص بك
try:
    # هنا قمت بالاستدعاء من الملف الذي حددته أنت
    from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT
    print("--- [AVATAR] Special Avatar Styles loaded successfully ---")
except ImportError:
    print("--- [CRITICAL ERROR] avatar_styles_config.py not found! ---")
    AVATAR_STYLES = {}
    AVATAR_NEGATIVE_PROMPT = ""

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt):
    try:
        # 1. معالجة الصورة (الزوم الذكي خلف الكواليس)
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        init_image = ImageOps.exif_transpose(init_image)

        # تحويل لمربع 1024 لضمان أعلى دقة لملامح الوجه
        init_image = ImageOps.fit(init_image, (1024, 1024), method=Image.LANCZOS, centering=(0.5, 0.4))

        # 2. جلب الستايل من قائمة الأفاتار حصراً
        # التأكد من استخدام AVATAR_STYLES كما هو في ملفك
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES.get("photorealistic", ""))
        
        # 3. دمج البرومبت
        final_prompt = f"{prompt}, {style_prompt}, masterpiece, sharp focus on face, detailed skin"

        # 4. إعدادات القوة (Strength) لضمان التغيير الفني مع الحفاظ على الشبه
        custom_strength = 0.60 if style_key != "photorealistic" else 0.45

        # 5. التوليد (Image-to-Image)
        torch.cuda.empty_cache()
        output_image = img_pipe(
            prompt=final_prompt,
            negative_prompt=AVATAR_NEGATIVE_PROMPT,
            image=init_image,
            strength=custom_strength,
            num_inference_steps=35,
            guidance_scale=10.0
        ).images[0]

        gc.collect()
        return output_image

    except Exception as e:
        print(f"--- [AVATAR GENERATOR ERROR]: {str(e)} ---")
        return init_image
