import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://s2-g1.glbimg.com/6uZR0zuQMMG_fdiBVKtbU6YYuQI=/0x309:1125x1500/984x0/smart/filters:strip_icc()/i.s3.glbimg.com/v1/AUTH_59edd422c0c84a879bd37670ae4f538a/internal_photos/bs/2020/T/G/zuoCPBSFq9HNdDeLjNFg/000-1ss8w5.jpg"

init_image = load_image(url).convert("RGB")
prompt = "a photo of a 20 years old female with blonde hair and green eyes"
image = pipe(prompt, image=init_image).images[0]
image.save("madmccann-im2im-20yo.png")