# dimensions_config.py

# هذه المقاسات هي الأفضل تقنياً لنموذج SDXL لضمان عدم وجود رؤوس مكررة أو تشويه
DIMENSIONS_MAP = {
    # المقاس المربع (انستجرام)
    "square": {"width": 1024, "height": 1024},
    "1:1": {"width": 1024, "height": 1024},

    # مقاس التيك توك والريلز (الطولي الكامل)
    "tiktok": {"width": 768, "height": 1344},
    "9:16": {"width": 768, "height": 1344},

    # مقاس البورتريه الاحترافي
    "portrait": {"width": 832, "height": 1216},
    "2:3": {"width": 832, "height": 1216},

    # مقاس العرض السينمائي (يوتيوب / أفلام)
    "landscape": {"width": 1216, "height": 832},
    "16:9": {"width": 1344, "height": 768},
    "3:2": {"width": 1216, "height": 832}
}

def get_image_dimensions(job_input):
    """
    تحديد المقاسات باحترافية مع دعم الأسماء والأرقام
    """
    # جلب القيمة من المدخلات، وإذا لم توجد نستخدم المربع كافتراضي
    ratio = str(job_input.get('aspect_ratio', 'square')).lower().strip()
    
    # البحث في الخريطة الاحترافية
    if ratio in DIMENSIONS_MAP:
        dim = DIMENSIONS_MAP[ratio]
        return dim["width"], dim["height"]
    
    # في حال أرسل المستخدم قيمة غير معرفة، نعود للمقاس القياسي
    print(f"--- [INFO] Aspect ratio '{ratio}' not found, using 1024x1024 ---")
    return 1024, 1024
