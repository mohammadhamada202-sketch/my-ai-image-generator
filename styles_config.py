# styles_config.py

STYLE_PROMPTS = {
    # الستايل الكرتوني: شكل ديزني/بيكسار مع ألوان زاهية وتعبيرات واضحة
    "cartoon": "highly stylized cartoon illustration, 3D Disney Pixar style, cute character design, bold expressive eyes, smooth textures, vibrant saturated colors, masterpiece, playful aesthetic",
    
    # الستايل الواقعي المعتمد في الموقع
    "realistic": "hyper-realistic portrait, 8k raw photo, cinematic lighting, ultra-detailed skin texture, professional studio photography[span_0](start_span)[span_0](end_span)",
    
    # ستايل الأنمي: رسم ياباني ثنائي الأبعاد
    "anime": "official anime style art, 2D, flat color, cel shaded, thick outlines, studio ghibli aesthetic, hand-drawn illustration, high-quality vector art, vibrant colors, no realism",
    
    # ستايل الألعاب 
    "3d_render": "highly detailed 3D gaming character, Unreal Engine 5 render, Octane render, stylized character design, smooth clay texture, cinematic gaming lighting",
    
    # ستايل البيكسل 
    "pixel_art": "genuine 8-bit pixel art, visible square pixels, nostalgic gaming aesthetic, clean pixel edges, 2D sprite",
    
    # ستايل السكيتش 
    "sketch": "raw charcoal sketch on textured paper, messy graphite pencil strokes, hand-drawn artistic lines, rough hatching, elegant minimalist portrait sketch",
    
    # ستايل التجريدي 
    "abstract": "abstract digital art, geometric shapes, double exposure, vibrant neon color splashes, liquid metal textures, artistic distortion",
    
    # زيادة احتياط لعدم حدوث خطأ
    "photorealistic": "professional studio portrait, 8k raw photo, cinematic rim lighting, masterpiece[span_1](start_span)[span_1](end_span)"
}

NEGATIVE_PROMPT = (
    "photorealistic, real life, 3d, photography, depth of field, blurry, distorted, "
    "lowres, text, watermark, bad anatomy, poorly drawn face, grainy, noisy, messy"
)
