import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    try:
        # 1. تحويل الـ Base64 إلى صورة
        init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        
        # 2. القص الذكي (Smart Square Crop)
        # هذا السطر يحل مشكلة الضخامة والتمطيط عبر قص الصورة من المنتصف لتصبح مربعة مثالية
        # بدلاً من ضغط الصورة الطولية، سيتم قصها لتناسب أبعاد الموديل (1024x1024) بشكل متناسق
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.5))
        
        # 3. هندسة البرومبت (لتحسين ستايل الأنمي والواقعية)
        face_enhancer = "highly detailed cinematic portrait, sharp focus, masterfully rendered, clean facial features"
        final_prompt = f"{style_prompt}, {prompt}, {face_enhancer}"
        
        # 4. عملية التوليد بإعدادات متوازنة
        # استخدام strength=0.5 يسمح بتطبيق ستايل فني (مثل الأنمي) بشكل أقوى مع الحفاظ على ملامحك الأساسية
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.5, 
            num_inference_steps=35,
            guidance_scale=9.0, # زيادة الالتزام بالستايل المختار لضمان نتيجة فنية واضحة
        ).images[0]
        
        return image
    except Exception as e:
        print(f"Avatar Generation Error: {str(e)}")
        # في حال حدوث خطأ، يتم إرجاع الصورة الأصلية لضمان عدم توقف السيرفر
        return init_image
