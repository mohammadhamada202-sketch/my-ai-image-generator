from google import genai
from google.genai import types # استيراد الأنواع للتحكم بالإعدادات
from PIL import Image
import base64
import io
import os

# [1] ستايلات مطورة لضمان أقصى دقة (Ultra-High Resolution)
AVATAR_STYLES = {
    "photorealistic": (
        "ultra-detailed professional cinematic portrait, hyper-realistic skin texture, "
        "shot on 85mm lens, f/1.4, sharp focus on eyes, soft bokeh background, "
        "8k raw photo, high-end studio lighting, subsurface scattering"
    ),
    "anime": (
        "masterpiece, official anime art style, studio ghibli aesthetic, "
        "high-quality 2D cel shaded, clean lineart, vibrant cinematic colors, "
        "expressive eyes, highly detailed background"
    ),
    "3d_render": (
        "Disney Pixar style 3D avatar, highly detailed digital character, "
        "Unreal Engine 5 render, cinematic lighting, smooth clay textures, "
        "masterfully rendered 3D art, vibrant colors"
    ),
    "pixel_art": (
        "genuine 8-bit pixel art, high-quality retro game sprite, "
        "clean square pixels, sharp edges, recognizable facial features, "
        "vibrant limited color palette"
    ),
    "sketch": (
        "fine charcoal sketch on textured paper, artistic graphite pencil strokes, "
        "hand-drawn minimalist portrait, high contrast black and white, "
        "elegant artistic hatching"
    ),
    "abstract": (
        "modern abstract digital art portrait, geometric shapes, double exposure, "
        "vibrant neon color splashes, artistic distortion, surreal masterpiece"
    )
}

def generate_avatar(image_b64, prompt, style_key):
    try:
        # [2] إعداد العميل (Client)
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # اختيار الستايل أو العودة للواقعي كافتراضي
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        
        # [3] هندسة الأمر النهائي لضمان أعلى جودة
        full_instruction = (
            f"Transform the person in this image into {style_prompt}. "
            f"Context: {prompt}. "
            "IMPORTANT: Maintain the person's identity, facial features, and gender. "
            "Output must be a high-fidelity image."
        )

        # [4] التوليد باستخدام الموديل الأقوى (Imagen 3)
        # ملاحظة: Imagen 3 هو الأفضل للصور الواقعية والتحويل الاحترافي
        response = client.models.generate_image(
            model='imagen-3.0-generate-002', # الإصدار المستقر والأقوى
            prompt=full_instruction,
            config=types.GenerateImageConfig(
                number_of_images=1,
                output_mime_type='image/png',
                add_watermark=False, # اختياري
                aspect_ratio="1:1"   # الأفضل للأفاتار
            )
        )

        # [5] استخراج الصورة وحفظها كـ PIL Image
        # المكتبة تعيد الصورة جاهزة في generated_images
        return response.generated_images[0].image

    except Exception as e:
        print(f"--- [AVATAR GENERATOR ERROR]: {str(e)} ---")
        # في حال الخطأ نعود بالصورة الأصلية لضمان عدم توقف السيرفر
        try:
            image_data = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(image_data))
        except:
            return None
