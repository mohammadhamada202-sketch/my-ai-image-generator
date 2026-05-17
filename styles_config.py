# styles_config.py

# خريطة الأنماط والموديلات الذكية لـ SmartGenAI (موجهة بالكامل لـ Google Vertex AI)
STYLE_CONFIGS = {
    # 1. الواقعي الخرافي
    "photorealistic": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "highly detailed 8k raw photo, cinematic corporate portrait, shot on 85mm lens, sharp focus, natural skin texture, realistic lighting, masterpiece"
    },
    
    # 2. نمط كرتون غربي نظيف
    "cartoon": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "cute 2D western cartoon style, adorable character design, vibrant soft colors, smooth digital vector art, clean distinct ink outlines, happy friendly aesthetic, storybook illustration, children book graphics, simple flat shading, NO anime, NO realism, masterpiece"
    },
    
    # 3. أنمي ياباني احترافي
    "anime": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "authentic Japanese anime style, 1990s anime aesthetic, sharp cinematic lineart, vivid anime color grading, masterpiece"
    },
    
    # 4. نمط بيكسار ثلاثي الأبعاد
    "pixar": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "official Disney Pixar 3D animation character style, smooth round features, detailed hair strands, playful studio lighting, cute 3D asset"
    },
    
    # 5. بيكسل آرت ريترو
    "pixel_art": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "meticulous 8-bit pixel art style, retro video game aesthetic, sharp vibrant pixels, distinct square coloring, masterpiece"
    },
    
    # 6. رسم يدوي (سكتش)
    "sketch": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "professional charcoal drawing style, hand-drawn pencil strokes, artistic hatching, high contrast, textured paper aesthetic, masterpiece"
    },
    
    # 7. ريندر ثلاثي الأبعاد
    "3d_render": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "highly detailed 3D character style, Unreal Engine 5 render style, ray tracing, polished textures, volumetric lighting, masterpiece"
    },
    
    # 8. تجريدي فني
    "abstract": {
        "provider": "google",
        "model": "imagen-3.0-generate-002",
        "prompt_enhancer": "abstract digital art portrait style, geometric shapes, double exposure effect, vibrant neon color splashes, artistic expression, masterpiece"
    }
}

# البرومبت السلبي الموحد لضمان الجودة
AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry, grainy, deformed face, unrecognizable, watermark"
