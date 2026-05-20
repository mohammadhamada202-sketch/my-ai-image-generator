import os
import io
import requests
import base64
import time
import runpod
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from supabase import create_client

# إعداد الستايلات (يمكنك مستقبلاً تحويلها لـ JSON)
AVATAR_STYLES = {
    "photorealistic": "professional cinematic portrait of the person, ultra-detailed eyes, sharp focus on face, 85mm lens, f/1.8, high-end studio lighting, 8k raw photo, extreme skin detail.",
    "anime": "masterpiece, official anime style, studio ghibli aesthetic, cel shaded, clean lineart, vibrant colors.",
    "3d_render": "Disney Pixar style 3D avatar, Unreal Engine 5 render, cinematic lighting, smooth clay textures."
}

GLOBAL_NEGATIVE_PROMPT = "futuristic, sci-fi, glowing particles, distorted clouds, abstract sky, fantasy elements, cartoonish, low resolution, blurry, deformed face, extra digits, watermark"

# إعداد MediaPipe مرة واحدة
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def process_image(img_data):
    """الزووم التلقائي على الوجه وإزالة الخلفية"""
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # 1. اكتشاف الوجه
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detections:
        raise Exception("No face detected in the image.")
    
    # الحصول على إحداثيات الوجه
    bbox = results.detections[0].location_data.relative_bounding_box
    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
    width, height = int(bbox.width * w), int(bbox.height * h)
    
    # الزووم (إضافة هامش)
    padding = 60
    y1, y2 = max(0, y - padding), min(h, y + height + padding)
    x1, x2 = max(0, x - padding), min(w, x + width + padding)
    cropped_img = img[y1:y2, x1:x2]
    
    # 2. إزالة الخلفية
    img_no_bg = remove(cropped_img)
    _, buffer = cv2.imencode('.png', img_no_bg)
    return buffer.tobytes()

def handler(job):
    try:
        print("--- [START] SMARTGEN - AVATAR PIPELINE V2.0 ---")
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        style_key = job_input.get('style', 'photorealistic')

        if not image_url:
            return {"error": "image_url is missing"}

        # 1. المعالجة المسبقة (زووم + تنظيف)
        img_raw_data = requests.get(image_url).content
        processed_img_bytes = process_image(img_raw_data)
        
        # تحويل لـ Base64 للـ API
        img_b64 = base64.b64encode(processed_img_bytes).decode('utf-8')

        # 2. إعداد Vertex AI
        aiplatform.init(project=os.environ.get("GCP_PROJECT"), location="us-central1")
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        
        # 3. صياغة البرومبت النهائي
        style_prompt = AVATAR_STYLES.get(style_key, AVATAR_STYLES["photorealistic"])
        full_prompt = f"{style_prompt}. Focus on the subject's face. The environment must be natural and neutral. NO sci-fi or abstract elements."

        # 4. التوليد
        response = model.generate_images(
            prompt=full_prompt,
            negative_prompt=GLOBAL_NEGATIVE_PROMPT,
            number_of_images=1,
            input_image=img_b64
        )

        generated_img = response.images[0]._image_bytes

        # 5. الرفع لـ Supabase
        sb = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        file_name = f"avatar_{int(time.time())}.png"
        sb.storage.from_("MyFirstImagesTest1").upload(file_name, generated_img)
        
        return {"avatar_url": sb.storage.from_("MyFirstImagesTest1").get_public_url(file_name)}

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
