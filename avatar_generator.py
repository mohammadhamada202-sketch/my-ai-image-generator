from google import genai
from google.genai import types
from PIL import Image
import base64
import io
import os

# ستايلات واقعية جداً تحاكي عدسات الكاميرا الحقيقية
AVATAR_STYLES = {
    "photorealistic": (
        "ultra-detailed professional cinematic portrait, hyper-realistic skin texture, "
        "shot on 85mm lens, f/1.4, sharp focus on eyes, soft bokeh background, "
        "8k raw photo, high-end studio lighting"
    ),
    "anime": "official anime art style, studio ghibli aesthetic, clean lineart, vibrant colors",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render, cinematic lighting",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, sharp pixels",
    "sketch": "fine charcoal sketch, artistic pencil strokes, high contrast B&W",
    "abstract": "abstract digital art portrait, neon color splashes, surreal masterpiece"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. Maintain identity and facial features."
        )

        # تحويل الصورة لـ Part متوافق مع المكتبة
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_b64),
            mime_type="image/png"
        )

        # التوليد باستخدام Imagen 3 الأقوى
        response = client.models.generate_content(
            model='imagen-3.0-generate-002',
            contents=[full_instruction, image_part]
        )

        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        # العودة بالصورة الأصلية في حال فشل الـ API
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
