# avatar_generator.py
import torch
from PIL import Image
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    # تحويل Base64 إلى PIL Image
    init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    init_image = init_image.resize((1024, 1024))
    
    final_prompt = f"{style_prompt}, {prompt}"
    
    image = img_pipe(
        prompt=final_prompt,
        image=init_image,
        strength=0.5, # درجة الحفاظ على ملامح الشخص
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt=negative_prompt
    ).images[0]
    
    return image