import torch
from PIL import Image, ImageOps, ImageFilter
import base64
from io import BytesIO
from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt):
    try:
        # 1. تحويل الصورة وتجهيزها بأعلى جودة
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 2. القص الذكي فائق الدقة (Ultra-Sharp Crop)
        # نستخدم أبعاد 1024x1024 وهي الدقة الأصلية لـ SDXL لضمان عدم ضياع أي بكسل
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.4), method=Image.LANCZOS)
        
        # 3. جلب الستايل وتعزيزه بكلمات "الدقة الفائقة"
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES.get("photorealistic"))
        
        # 4. إضافة "محفزات الدقة" (Quality Boosters)
        # هذه الكلمات تجبر الموديل على معالجة التفاصيل الدقيقة جداً في المسام والعيون والشعر
        quality_boosters = (
            "8k resolution, highly defined features, extremely detailed eyes, "
            "sharp focus, professional lighting, masterpiece, intricate details, "
            "hyper-detailed skin texture, subsurface scattering, stunning visual effects"
        )
        
        final_prompt = f"{style_prompt}, {quality_boosters}, maintaining exact identity"
        
        # 5. التحكم بالدقة عبر خطوات التوليد (Inference Steps)
        # رفعنا الـ steps لـ 50 لضمان تنظيف الصورة من أي "غباش" أو نويز
        # الـ guidance_scale المرتفع (15.0) يضمن حِدّة الخطوط (Sharpness)
        
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=AVATAR_NEGATIVE_PROMPT + ", blurry, low resolution, grainy, fuzzy, soft focus",
            image=init_image,
            strength=0.52, # توازن مثالي للحفاظ على الشبه مع الدقة العالية[cite: 1]
            num_inference_steps=50, 
            guidance_scale=15.0,
            target_size=(1024, 1024),
            original_size=(1024, 1024)
        ).images[0]
        
        # 6. لمسة التحسين النهائية (Post-Processing)
        # إضافة فلتر شحذ خفيف لزيادة وضوح الملامح
        image = image.filter(ImageFilter.SHARPEN)
        
        return image

    except Exception as e:
        print(f"Post-Update Error: {str(e)}")
        return init_image
