from google import genai
from PIL import Image
import base64
import io
import os

# ستايلات واقعية جداً تحاكي التصوير الفوتوغرافي الحقيقي
AVATAR_STYLES = {
    "photorealistic": (
        "ultra-detailed professional cinematic portrait, hyper-realistic skin texture, "
        "shot on 85mm lens, f/1.4, sharp focus on eyes, 8k raw photo"
    ),
    "anime": "high-quality anime style, studio ghibli aesthetic, vibrant cinematic colors",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render, cinematic lighting",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, clean pixels",
    "sketch": "fine charcoal sketch, artistic pencil strokes, high contrast",
    "abstract": "abstract digital art portrait, neon splashes, surreal masterpiece"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. Maintain identity and facial features."
        )

        # تحويل الصورة واستخدام كائن images للتوليد
        response = client.images.generate(
            model='imagen-3.0-generate-002',
            prompt=full_instruction,
            image=base64.b64decode(image_b64)
        )

        # استعادة الصورة من بيانات البتات
        image_bits = response.generated_images[0].image.bits
        return Image.open(io.BytesIO(image_bits))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        # العودة بالصورة الأصلية في حال حدوث أي خطأ
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
