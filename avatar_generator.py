from google import genai
from PIL import Image
import base64
from io import BytesIO
import os

# الستايلات المدمجة
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait, ultra-detailed eyes, 8k raw photo",
    "anime": "official anime style art, studio ghibli aesthetic, clean lineart",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite",
    "sketch": "raw charcoal sketch, hand-drawn artistic lines, high contrast",
    "abstract": "abstract digital art, vibrant neon splashes, artistic distortion"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        # استخدام المكتبة الجديدة
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # هندسة الأمر
        full_instruction = (
            f"Transform the person in this image into this style: {style_prompt}. "
            f"Context: {prompt}. Maintain identity and gender."
        )

        # التوليد باستخدام Nano Banana 2 (Gemini 3 Flash Image)
        response = client.models.generate_image(
            model='gemini-3-flash-image',
            prompt=full_instruction,
            image=image_b64
        )

        return Image.open(BytesIO(response.image.bits))

    except Exception as e:
        print(f"--- [AVATAR ERROR]: {str(e)} ---")
        image_data = base64.b64decode(image_b64)
        return Image.open(BytesIO(image_data))
