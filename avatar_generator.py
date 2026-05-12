from google import genai
from PIL import Image
import base64
import io
import os

AVATAR_STYLES = {
    "photorealistic": "ultra-detailed professional cinematic portrait, 8k raw photo, hyper-realistic skin",
    "anime": "high-quality anime style, vibrant cinematic colors, studio ghibli aesthetic",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render",
    "pixel_art": "8-bit pixel art, retro video game sprite",
    "sketch": "fine charcoal sketch, artistic pencil strokes",
    "abstract": "abstract digital art portrait, neon splashes"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        full_instruction = f"Transform the person in this image into {style_prompt}. Context: {prompt}. Maintain identity."
        
        image_part = {"mime_type": "image/png", "data": image_b64}
        
        # تجربة الموديلات المتاحة
        potential_models = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest']
        response = None

        for model_name in potential_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[full_instruction, image_part]
                )
                if response: break
            except: continue

        if not response:
            raise Exception("Avatar generation failed on all available models.")

        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
