import torch
import base64
import logging
from io import BytesIO
from PIL import Image
from diffusers import StableVideoDiffusionPipeline

logger = logging.getLogger(__name__)
VIDEO_CACHE_DIR = "/workspace/models/video"

def generate_video_logic(job):
    try:
        job_input = job['input']
        image_base64 = job_input.get('image_base64')
        
        if not image_base64:
            return {"error": "Missing image for video mode"}

        # معالجة الصورة
        image_data = base64.b64decode(image_base64)
        input_image = Image.open(BytesIO(image_data)).convert("RGB")
        input_image = input_image.resize((1024, 576))

        # تحميل موديل الفيديو SVD
        logger.info("Loading Video Model...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=VIDEO_CACHE_DIR
        ).to("cuda")

        # توليد الفيديو
        logger.info("Generating Video Frames...")
        generator = torch.manual_seed(42)
        frames = pipe(input_image, decode_chunk_size=8, generator=generator).frames[0]
        
        # تحويل أول فريم كـ Base64 (للتجربة)
        buffered = BytesIO()
        frames[0].save(buffered, format="PNG")
        
        return {
            "video_base64": base64.b64encode(buffered.getvalue()).decode("utf-8"),
            "message": "Video generated successfully (Sample Frame returned)"
        }
    except Exception as e:
        logger.error(f"Video Error: {e}")
        return {"error": str(e)}
