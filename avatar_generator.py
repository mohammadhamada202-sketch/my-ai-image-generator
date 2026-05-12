from google import genai
from PIL import Image
import base64
import io
import os

# ستايلات مصممة لتعطي نتائج "ترى بالعين" من حيث الواقعية
AVATAR_STYLES = {
    "photorealistic": (
        "ultra-detailed professional cinematic portrait, hyper-realistic skin texture, "
        "8k raw photo, cinematic lighting, masterpiece, shot on 85mm lens"
    ),
    "anime": "high-quality anime style, studio ghibli aesthetic, vibrant cinematic colors",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render, cinematic lighting",
    "pixel_art": "genuine 8-bit pixel art, retro video game sprite, clean pixels",
    "sketch": "fine charcoal sketch, artistic pencil strokes, high contrast",
    "abstract": "abstract digital art portrait, neon splashes, surreal masterpiece"
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. Maintain identity and facial features."
        )
        
        # تحضير الصورة كـ Inline Data للموديل
        image_part = {"mime_type": "image/png", "data": image_b64}

        # طلب التحويل من Gemini
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[full_instruction, image_part]
        )

        img_raw = response.candidates[0].content.parts[0].inline_data.data
        return Image.open(io.BytesIO(img_raw))

    except Exception as e:
        print(f"--- [GENERATOR ERROR] ---: {str(e)}")
        # في حال حدوث خطأ، نعيد الصورة الأصلية لضمان عدم توقف الخدمة
        return Image.open(io.BytesIO(base64.b64decode(image_b64)))
