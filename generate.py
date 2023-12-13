import os
#pip install diffusers
from diffusers import StableDiffusionPipeline
import torch

#function to generate AI based images using Huggingface Diffusers
def generate_images_using_huggingface_diffusers(text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = text
    image = pipe(prompt).images[0]
    return image


result = generate_images_using_huggingface_diffusers("a beautiful girl laughing hard")
print(result)