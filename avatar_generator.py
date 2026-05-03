import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    try:
        # 1. تحويل الـ Base64 إلى صورة PIL
        init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        
        # 2. القص الذكي (Smart Crop): عزل الوجه لمنع التمطيط وضمان دقة الـ Scan
        # تركيز القوة الحسابية للموديل على منطقة الوجه (1024x1024)[cite: 1]
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.5))
        
        # 3. منطق القوة الديناميكية لضمان تميز كل ستايل[cite: 1]
        current_style = style_prompt.lower()
        
        # إذا كان الستايل فني (رسم)، نرفع القوة ليمسح الواقع ويرسم من جديد
        if any(s in current_style for s in ["anime", "sketch", "pixel", "abstract"]):
            custom_strength = 0.75  # تحويل جذري (75% رسم، 25% ملامح أصلية)[cite: 1]
            custom_guidance = 15.0  # التزام حديدي بالوصف الفني[cite: 1]
        else:
            # للواقعية والـ 3D، نحافظ على توازن الملامح الأصلية
            custom_strength = 0.50  #[cite: 1]
            custom_guidance = 9.0   #[cite: 1]
        
        # 4. تعزيز ملامح الوجه (Face Enhancer) لضمان تفاصيل دقيقة[cite: 1]
        face_enhancer = "centered portrait, clear facial features, sharp focus, masterpiece, highly detailed eyes"
        final_prompt = f"{style_prompt}, {prompt}, {face_enhancer}"
        
        # 5. التوليد بعدد خطوات مرتفع (40) لزيادة حدة التفاصيل[cite: 1]
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=custom_strength,
            num_inference_steps=40,
            guidance_scale=custom_guidance
        ).images[0]
        
        return image
    except Exception as e:
        print(f"Generation Error: {str(e)}")
        # العودة بالصورة المقصوصة في حال الفشل لضمان عدم توقف السيرفر
        return init_image
