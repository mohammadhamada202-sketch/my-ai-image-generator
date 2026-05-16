# styles_config.py

# خريطة الأنماط والموديلات الذكية لـ SmartGenAI
STYLE_CONFIGS = {
    # 1. الواقعي
    "photorealistic": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "highly detailed 8k raw photo, cinematic corporate portrait, shot on 85mm lens, sharp focus, natural skin texture, realistic lighting, masterpiece"
    },
    
    # 2. كرتون كلاسيكي مسطح (تغيير شامل لإلغاء أي طابع أنمي أو واقعي)
    "cartoon": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "vibrant 1990s 2D flat cartoon style, vintage western animation cell art, Hanna-Barbera and early Disney aesthetic, solid bold block colors, thick black outlines, minimal flat shading, absolutely NO depth, NO textures, NO realistic lighting, NO detailed render, clean classic vector graphic, masterpiece"
    },
    
    # 3. أنمي
    "anime": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "authentic Japanese anime style, 1990s anime aesthetic, sharp cinematic lineart, vivid anime color grading, masterpiece"
    },
    
    # 4. نمط بيكسار
    "pixar": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "official Disney Pixar 3D animation character style, smooth round features, detailed hair strands, playful studio lighting, cute 3D asset"
    },
    
    # 5. بيكسل آرت
    "pixel_art": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "meticulous 8-bit pixel art, retro video game aesthetic, sharp vibrant pixels, masterpiece"
    },
    
    # 6. رسم يدوي (سكتش)
    "sketch": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "professional charcoal drawing, hand-drawn strokes, artistic hatching, high contrast, textured paper aesthetic, masterpiece"
    },
    
    # 7. ريندر ثلاثي الأبعاد
    "3d_render": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "highly detailed 3D character, Unreal Engine 5 render, ray tracing, polished textures, volumetric lighting, masterpiece"
    },
    
    # 8. تجريدي
    "abstract": {
        "provider": "together",
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt_enhancer": "abstract digital art portrait, geometric shapes, double exposure, vibrant neon color splashes, artistic expression, masterpiece"
    }
}

# البرومبت السلبي الموحد لضمان الجودة
AVATAR_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry, grainy, deformed face, unrecognizable, watermark"
