import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://f.rpp-noticias.io/2021/06/28/1113348e46dxlcxoai3tp7jpg.jpg"

init_image = load_image(url).convert("RGB")
prompt = "a photo of a 30 years old male caucasian guy, similar to Mark Zuckerberg with dark blonde hair, brown eyes, slim face, and no beard. He is smiling and looking at the camera with his eyes"
image = pipe(prompt, image=init_image).images[0]
image.save("sketchedface.png")