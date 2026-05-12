from google import genai
from PIL import Image
import base64
import io
import os

AVATAR_STYLES = {
    "photorealistic": "ultra-detailed professional cinematic portrait, 8k raw photo, realistic skin",
    "anime": "official anime style, studio ghibli aesthetic, clean lineart",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render",
    "pixel_art": "genuine 8-bit pixel art, sharp pixels",
    "sketch": "fine charcoal sketch, high contrast",
    "abstract": "abstract digital art, neon splashes"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        full_instruction = f"Transform the person in this image into {style_prompt}. Context: {prompt}"

        # استخدام دالة imagen لضمان أعلى واقعية للوجه
        response = client.models.imagen(
            model='imagen-3.0-generate-002',
            prompt=full_instruction,
            image=base64.b64decode(image_b64) # إرسال الصورة الخام للتحويل
        )

        return Image.open(io.BytesIO(response.generated_images[0].image.bits))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
