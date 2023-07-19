import torch
import numpy as np
import PIL
from PIL import Image
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.data import shanghai

def imageloader(image, size=None):
    if not image.mode == "RGB": 
        image = image.convert("RGB")
    
    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
        (w - crop) // 2:(w + crop) // 2]

    image2 = Image.fromarray(img)
    if size is not None:
        image2 = image2.resize((size, size), resample=PIL.Image.BICUBIC)

    image2 = np.array(image2).astype(np.uint8)

    example = (image2 / 127.5 - 1.0).astype(np.float32)

    return example


def ldm_cond_sample(config_path, ckpt_path, batch_size, name):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x = next(iter(dataloader))
    seg = x['rgb']
    # seg = image
    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)

        samples = model.decode_first_stage(samples)

    save_image(condition, out_name + '_cond.png')
    save_image(samples, out_name + '_sample.png')


if __name__ == '__main__':

    out_name = "2023-05-24T21-05-53_shanghai_5000"
    config_path = 'configs/latent-diffusion/shanghai_5000.yaml' # origin: crossattn, 3: concate
    
    ckpt_path = 'logs/' + out_name + '/checkpoints/epoch=000371.ckpt' # concate
    
    # ckpt_path = 'logs/2023-05-09T20-17-06_shanghai/checkpoints/last.ckpt' 

    # inputimg = Image.open('test1.jpg')
    # image = imageloader(inputimg, 256)
    dataset = shanghai.ShanghaiTest(size=512)
    ldm_cond_sample(config_path, ckpt_path, batch_size=1, name = out_name)

    # ldm_cond_sample(config_path, ckpt_path, 1, image)
