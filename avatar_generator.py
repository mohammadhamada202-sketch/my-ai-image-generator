from google import genai
from PIL import Image
import base64
import io
import os

AVATAR_STYLES = {
    "photorealistic": "ultra-detailed cinematic portrait, 8k raw photo, realistic skin texture",
    "anime": "high-quality anime style, studio ghibli aesthetic, vibrant colors",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render",
    "pixel_art": "8-bit pixel art, retro video game sprite",
    "sketch": "charcoal sketch, artistic pencil strokes",
    "abstract": "abstract digital art, neon splashes"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        full_instruction = f"Transform the person in this image into {style_prompt}. Context: {prompt}"
        
        # تحضير الصورة كجزء من المحتوى
        image_part = {"mime_type": "image/png", "data": image_b64}

        # استخدام generate_content لضمان التوافق
        response = client.models.generate_content(
            model='imagen-3.0-generate-002',
            contents=[full_instruction, image_part]
        )

        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
