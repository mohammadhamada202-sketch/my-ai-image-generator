import torch
from PIL import Image
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    try:
        # 1. تحويل الـ Base64 إلى صورة PIL
        init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        
        # 2. استخراج المقاسات الأصلية (للحفاظ على نسبة الطول للعرض)
        orig_width, orig_height = init_image.size
        
        # التأكد من أن المقاسات من مضاعفات 8 لضمان استقرار موديل SDXL
        width = (orig_width // 8) * 8
        height = (orig_height // 8) * 8
        
        # إعادة تحجيم الصورة لتناسب الحسابات البرمجية بدقة
        init_image = init_image.resize((width, height), Image.LANCZOS)
        
        # 3. دمج الأوصاف (الستايل + وصف الوجه)
        face_details = "highly detailed face, realistic eyes, cinematic lighting, masterpiece, 8k"
        final_prompt = f"{style_prompt}, {prompt}, {face_details}"
        
        # 4. عملية التوليد بنفس الأبعاد الأصلية
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.45,       # يحافظ على ملامحك بنسبة 55% ويغير الستايل 45%
            width=width,
            height=height,
            num_inference_steps=35,
            guidance_scale=8.0
        ).images[0]
        
        return image
    except Exception as e:
        print(f"Error in avatar_generator: {str(e)}")
        # العودة بصورة فارغة أو الصورة الأصلية في حال الفشل
        return init_image
