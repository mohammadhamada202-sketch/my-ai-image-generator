import torch
from PIL import Image
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    # 1. تحويل الـ Base64 ومعالجة الصورة الأصلية
    init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    
    # تحجيم الصورة لمقاس SDXL المثالي
    init_image = init_image.resize((1024, 1024), Image.LANCZOS)
    
    # 2. هندسة البرومبت ليكون واقعياً (Photorealistic Avatar)
    # إضافة كلمات مفتاحية لتعزيز تفاصيل الوجه والواقعية
    face_details = "highly detailed face, sharp eyes, cinematic lighting, hyper-realistic skin texture, 8k resolution, masterpiece"
    final_prompt = f"{style_prompt}, {prompt}, {face_details}"
    
    # 3. إعدادات الـ Face Scanning والحفاظ على الملامح
    # القيمة 0.45 هي "نقطة التوازن" (Sweet Spot):
    # أقل من 0.4: تحافظ على الشخص جداً لكن لا يتغير الستايل.
    # أكثر من 0.6: يتغير الستايل تماماً وتضيع ملامح الشخص.
    STRENGTH = 0.45 
    
    # 4. عملية التوليد
    image = img_pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=STRENGTH,
        num_inference_steps=40, # زيادة الخطوات لزيادة الدقة والواقعية
        guidance_scale=8.5,     # زيادة الالتزام بالوصف
        eta=0.0,
    ).images[0]
    
    return image
