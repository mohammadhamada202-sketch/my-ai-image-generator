# text_generator.py
import torch

def generate_from_text(pipe, prompt, style_prompt, negative_prompt, width, height):
    final_prompt = f"{style_prompt}, {prompt}"
    
    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    return image