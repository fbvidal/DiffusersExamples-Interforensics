import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://f.rpp-noticias.io/2021/06/28/1113348e46dxlcxoai3tp7jpg.jpg"

init_image = load_image(url).convert("RGB")
prompt = "a real face photo of Mark Zuckerberg."
image = pipe(prompt, image=init_image).images[0]
image.save("sketchedface.png")