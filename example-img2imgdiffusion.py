import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://media.wired.com/photos/592676467034dc5f91beb80e/master/w_2240,c_limit/MarkZuckerberg.jpg"

init_image = load_image(url).convert("RGB")
prompt = "transform in a coloured person of Mark Zuckerberg."
image = pipe(prompt, image=init_image).images[0]
image.save("sketchedface.png")