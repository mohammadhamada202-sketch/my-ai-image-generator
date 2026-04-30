import torch
import base64
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# الستايلات الأساسية
STYLES = {
    "realistic": {"p": "professional photo, hyper-realistic, 8k", "def_s": 0.35},
    "anime": {"p": "anime style, studio ghibli, vibrant", "def_s": 0.6},
    "pixar": {"p": "disney pixar 3d style, cute, smooth", "def_s": 0.55},
    "cartoon": {"p": "cartoon illustration, bold outlines", "def_s": 0.7}
}

class ImageEngine:
    def __init__(self, model_path, cache_dir):
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir
        ).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def _convert_ui_strength(self, ui_value):
        """
        تحويل القيمة من (1-100) لتصبح بين (0.3-0.7)
        المعادلة: (القيمة / 250) + 0.3
        إذا كانت 1 -> تعطي تقريباً 0.3
        إذا كانت 100 -> تعطي 0.7
        """
        try:
            val = float(ui_value)
            # التأكد أن القيمة لا تخرج عن الحدود
            val = max(1, min(100, val)) 
            # تحويل المقياس (Linear Mapping)
            return 0.3 + (val - 1) * (0.7 - 0.3) / (100 - 1)
        except:
            return 0.5 # قيمة افتراضية في حال الخطأ

    def process(self, base64_image, prompt, style_key, ui_strength):
        # 1. جلب الستايل
        style = STYLES.get(style_key, STYLES["realistic"])
        full_prompt = f"{style['p']}, {prompt}"
        
        # 2. تحويل الـ Strength من شريط المستخدم (1-100) إلى (0.3-0.7)
        final_strength = self._convert_ui_strength(ui_strength)
        
        # 3. تحويل الصورة
        init_img = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB").resize((1024, 1024))
        
        output = self.pipe(
            prompt=full_prompt,
            image=init_img,
            strength=final_strength,
            negative_prompt="low quality, blurry, distorted, deformed",
            num_inference_steps=30
        ).images[0]

        # 4. التشفير لـ base64
        buf = BytesIO()
        output.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")