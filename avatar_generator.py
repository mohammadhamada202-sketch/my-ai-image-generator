import torch
from PIL import Image, ImageOps
import base64
from io import BytesIO

def generate_avatar(img_pipe, image_b64, prompt, style_prompt, negative_prompt):
    try:
        # تحويل Base64 إلى PIL Image
        init_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        
        # القص الذكي لتجنب التمطيط
        init_image = ImageOps.fit(init_image, (1024, 1024), centering=(0.5, 0.5))
        
        # تحسين الوصف
        face_enhancer = "highly detailed cinematic portrait, sharp focus, masterpiece, clean facial features"
        final_prompt = f"{style_prompt}, {prompt}, {face_enhancer}"
        
        # التوليد
        image = img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.5,
            num_inference_steps=35,
            guidance_scale=9.0
        ).images[0]
        
        return image
    except Exception as e:
        print(f"Error: {str(e)}")
        return Image.new('RGB', (1024, 1024), color = 'white')
