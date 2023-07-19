import torch
import numpy as np
import PIL
import os
from PIL import Image
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import cv2
from torchvision.utils import make_grid
from PIL import Image
from pytorch_lightning import seed_everything

@torch.no_grad()
def normalize_zero(img):
    #입력: z
    zero_one = (img - img.min())/(img.max()-img.min()) 
    return zero_one # 0~1 정규화

@torch.no_grad()
def normalize_minus(img):
    #입력: 0~1
    minus_one = 2.*img - 1
    
    return minus_one # -1~1 정규화

@torch.no_grad()
def save_img(img, batch_size):

    grid = make_grid(img)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    if batch_size!=1: # c h w
        ndarr = rearrange(ndarr, 'h w i -> h (i w)')
    else:
        # ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("x_start2.png", ndarr)
    return ndarr

@torch.no_grad()
def to_arr(img, batch_size):

    grid = make_grid(img)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    if batch_size!=1: # c h w
        ndarr = rearrange(ndarr, 'h w i -> h (i w)')
    else:
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("x_start2.png", ndarr)
    return ndarr

@torch.no_grad()
def to_gray(img):
    c1 = img[:,0,:,:].unsqueeze(0)
    c2 = img[:,1,:,:].unsqueeze(0)
    c3 = img[:,2,:,:].unsqueeze(0)
    gray = c2+c3 # 0~ 최대 2
    return gray # [ b 1 h w ]

@torch.no_grad()
def counting(image, batch_size, cond):
    # image = ((image / normalizer).clip(0,1)*255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    image=cv2.resize(image, (512*batch_size, 512))
    image = image[0:504,0:512*batch_size]
    cv2.imwrite("origin.png", image)
    image = cv2.bitwise_not(image)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                        cv2.THRESH_BINARY, 5, -1)

    erode = cv2.erode(th2, kernel, iterations = 1)

    morphImg = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(cond, contours, -1, (0,255,0), thickness=-1)
    cv2.imwrite("result.png", cond)

    return len(contours) # remove the outerboarder countour
    
class ShanghaiTestBase(Dataset):
    def __init__(self,
                 input_img,
                 size=None,
                 interpolation="bicubic"
                 ):
        self._length = 1
        self.img = input_img
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.cond_paths = [0]
        self. labels = {
            "cond_path_": [l for l in self.cond_paths],
        }
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.img
        if not image.mode == "RGB": 
            image = image.convert("RGB")    

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)        
        example['rgb'] = (image / 127.5 - 1.0).astype(np.float32)
            
        return example

class ShanghaiTest(ShanghaiTestBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def ldm_cond_sample(model, batch_size, cv2img):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = next(iter(dataloader))
    seg = x['rgb']
    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)
        save_img(condition, batch_size)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, intermediates = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                    ddim_steps=200, eta=1.)
        # save_image(samples, out_name + '_zsample.png')
        # samples = model.decode_first_stage(samples)
        # save_image(samples, out_name + '_sample.png')
        # print(len(intermediates['pred_x0']), len(intermediates['x_inter']))
        # save_image(intermediates['pred_x0'][6], "iter.png")
        pred_img = normalize_minus(to_gray(normalize_zero(samples)))
        pred_arr = to_arr(pred_img,batch_size)
        crowd_pred = counting(pred_arr,batch_size, cv2img)
        # error_list.append(abs((x['crowd'].sum() - crowd_pred)/x['crowd'].sum()))
    return crowd_pred

if __name__ == '__main__':
## 2023-06-07T03-02-27_shanghai_amh << 기록
    seed_everything(23)
    device = torch.device('cuda')
    config_path = 'logs/2023-06-09T19-25-50_spatial_concat_nodenoise/configs/2023-06-09T19-25-50-project.yaml' # origin: crossattn, 3: concate
    ckpt_path = 'logs/' + '2023-06-09T19-25-50_spatial_concat_nodenoise' + '/checkpoints/last.ckpt' # concate
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, device, None)
    root = "../train/train_data/train_img_amh/"
    list = os.listdir(root)
    list.sort()
    error_list = []
    i = 0
    for imgname in list:
        image_name = root+imgname
        gt_name = image_name.replace("IMG", "DENSITY").replace("img", "density")
        print(image_name, gt_name)

        gtimg =  cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
        crop = min(gtimg.shape[0], gtimg.shape[1])
        gtimgh, gtimgw, = gtimg.shape[0], gtimg.shape[1]
        gtimg = gtimg[(gtimgh - crop) // 2:(gtimgh + crop) // 2,
            (gtimgw - crop) // 2:(gtimgw + crop) // 2]

        inputimg = Image.open(image_name)
        cv2image = cv2.cvtColor(np.array(inputimg), cv2.COLOR_RGB2BGR)
        crop = min(cv2image.shape[0], cv2image.shape[1])
        h, w, = cv2image.shape[0], cv2image.shape[1]
        img = cv2image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
        cv2image=cv2.resize(img, (512, 512))
        
        dataset = ShanghaiTest(input_img=inputimg, size=256)#512
        
        crowd_pred = ldm_cond_sample(model, batch_size=1, cv2img = cv2image)
        print("num:",i, "img:", imgname, "gt:", np.sum(gtimg)/205, "pred:", crowd_pred, "error rate:", abs(crowd_pred-np.sum(gtimg)/205)/(np.sum(gtimg)/205))
        error_list.append(abs(crowd_pred-np.sum(gtimg)/205)/(np.sum(gtimg)/205))
        i+=1
    print("list(percents):",error_list)
    print("mean error_rate :", sum(error_list)/len(list))

    # ldm_cond_sample(config_path, ckpt_path, 1, image)
