import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from huggingface_hub import hf_hub_download

from ip_adapter import IPAdapterPlusXL

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

base_model_path = "cagliostrolab/animagine-xl-3.1"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip_adapter_plus_test_.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# load ip-adapter
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

# read image prompt
image = Image.open("assets/images/woman.png")
image.resize((512, 512))

# generate image variations with only image prompt
num_samples = 2
images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=42)
grid = image_grid(images, 1, num_samples)
grid


# multimodal prompts
images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=42,
        prompt="best quality, high quality, wearing sunglasses on the beach", scale=0.5)
grid = image_grid(images, 1, num_samples)
grid