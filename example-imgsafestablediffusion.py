import torch
from diffusers import StableDiffusionPipelineSafe

pipeline = StableDiffusionPipelineSafe.from_pretrained(
    "AIML-TUDA/stable-diffusion-safe", torch_dtype=torch.float16
)
prompt = "the four horsewomen of the apocalypse, painting by tom of finland, gaston bussiere, craig mullins, j. c. leyendecker"
image = pipeline(prompt=prompt, **SafetyConfig.MEDIUM).images[0]
image["images"][0].save("resultsafestablediff.jpg")