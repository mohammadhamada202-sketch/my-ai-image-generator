# avatar_styles_config.py

AVATAR_STYLES = {
    # 📸 Portrait Mode: تركيز عالي على الوجه مع خلفية مغبشة احترافية
    "photorealistic": "professional cinematic portrait of the person, ultra-detailed eyes, sharp focus on face, shot on 85mm lens, f/1.8, soft bokeh background, blurred backdrop, high-end studio lighting, 8k raw photo, extreme skin detail, subsurface scattering",
    
    # 🎌 Anime Style: تحسين الخطوط والألوان لتصبح كأنمي احترافي
    "anime": "masterpiece, official anime style art of the person, high-quality 2D, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors, highly detailed expressive eyes, anime character design, best quality",
    
    # 🎮 3D Render: ستايل بيكسار المطور
    "3d_render": "highly detailed 3D Disney Pixar style avatar of the person, stylized digital character, Unreal Engine 5 render, subsurface scattering, cinematic gaming lighting, smooth clay textures, masterfully rendered 3D art",
    
    # 👾 Pixel Art: بيكسل آرت نظيف وحاد
    "pixel_art": "genuine 8-bit pixel art avatar of the person, retro video game sprite, limited color palette, clean square pixels, sharp edges, recognizable facial features in pixel form",
    
    # ✏️ Sketch: رسم يدوي فخم
    "sketch": "raw charcoal sketch of the person on textured paper, messy graphite pencil strokes, hand-drawn artistic lines, rough hatching, elegant minimalist portrait, high contrast black and white",
    
    # 🎨 Abstract: فن تجريدي عصري
    "abstract": "abstract digital art portrait of the person, geometric shapes, double exposure, vibrant neon color splashes, artistic distortion, dreamlike surreal composition, masterpiece"
}

# تحسين البرومبت السلبي لمنع التداخل بين الستايلات
AVATAR_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, grainy, fuzzy, deformed face, unrecognizable"
)
