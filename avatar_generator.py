import torch
from PIL import Image, ImageOps, ImageFilter
import base64
from io import BytesIO
from avatar_styles_config import AVATAR_STYLES, AVATAR_NEGATIVE_PROMPT

def generate_avatar(img_pipe, image_b64, prompt, style_key, negative_prompt):
    try:
        # 1. تجهيز الصورة الأصلية بأفضل خوارزمية قص (LANCZOS) لضمان الدقة
        image_data = base64.b64decode(image_b64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.4), method=Image.LANCZOS)
        
        # 2. جلب الستايل المطور
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # 3. هندسة البرومبت لقفل الهوية (Identity Locking)
        identity_boost = "precise facial likeness, maintaining same identity, highly recognizable person, detailed facial features"
        final_prompt = f"{style_prompt}, {identity_boost}, sharp focus"

        # 4. ضبط القيم للتجارب المكثفة (High Fidelity Settings)
        # الـ Guidance Scale العالي يضمن الالتزام بستايل الـ Bokeh والغباش في الخلفية
        # الـ Steps العالية (50) تضمن صورة كريستالية بدون نويز
        custom_guidance = 15.0 
        custom_steps = 50
        
        # توازن الشبه: 0.50 هي النقطة السحرية بين الستايل والشبه[cite: 1]
        custom_strength = 0.50 if style_key != "photorealistic" else 0.45

        # 5. عملية التوليد
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=AVATAR_NEGATIVE_PROMPT,
            image=init_image,
            strength=custom_strength,
            num_inference_steps=custom_steps,
            guidance_scale=custom_guidance,
            target_size=(1024, 1024),
            original_size=(1024, 1024)
        ).images[0]
        
        # 6. لمسة التحسين النهائية (Post-Processing) لزيادة الحِدّة[cite: 1]
        image = image.filter(ImageFilter.SHARPEN)
        
        return image

    except Exception as e:
        print(f"Update Error: {str(e)}")
        return init_image
