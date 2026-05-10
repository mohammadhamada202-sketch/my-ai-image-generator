# dimensions_config.py

# تعريف المقاسات القياسية لـ SDXL لضمان أعلى جودة ودقة
DIMENSIONS_MAP = {
    "square": {"width": 1024, "height": 1024},      # 1:1 (إنستجرام)
    "portrait": {"width": 832, "height": 1216},    # 2:3 (بورتريه احترافي)
    "tiktok": {"width": 768, "height": 1344},      # 9:16 (تيك توك، ريلز، ستوري)
    "landscape": {"width": 1216, "height": 832},   # 3:2 (سينمائي/عرضي)
    "standard": {"width": 1024, "height": 1024}     # الافتراضي
}

def get_image_dimensions(job_input):
    """
    تحديد المقاسات بناءً على نسبة العرض (aspect_ratio) المرسلة من التطبيق.
    """
    aspect_ratio = job_input.get('aspect_ratio', 'square').lower()
    
    # التحقق من وجود المقاس في القائمة
    if aspect_ratio in DIMENSIONS_MAP:
        return DIMENSIONS_MAP[aspect_ratio]["width"], DIMENSIONS_MAP[aspect_ratio]["height"]
    
    # مقاس احتياطي في حال عدم إرسال قيمة صحيحة
    return 1024, 1024
