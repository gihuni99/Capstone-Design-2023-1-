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
import math

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
        a = 1
        # cv2.imwrite("x_start2.png", ndarr)
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
def counting(image, batch_size, cond, save_name):
    # image = ((image / normalizer).clip(0,1)*255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    image=cv2.resize(image, (512*batch_size, 512))
    # image = image[0:504,0:512*batch_size]
    # cv2.imwrite('outputs/origins.png', image)
    # image = cv2.bitwise_not(image)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                        cv2.THRESH_BINARY, 5, -1)
    # cv2.imwrite('outputs/adaptiveThreshold.png', th2)

    # erode = cv2.erode(th2, kernel, iterations = 1)
    # cv2.imwrite('outputs/erode.png', erode)

    morphImg = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('outputs/morphImg.png', morphImg)
    contours, _ = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(cond, contours, -1, (0,255,0), thickness=3)
    ##################################################################3
    text="Prediction: " + str(len(contours))
    org=(10,500)
    font=cv2.FONT_HERSHEY_SIMPLEX
    size, BaseLine=cv2.getTextSize(text,font,1,2)
    cv2.rectangle(cond,org,(org[0]+size[0],org[1]-size[1]),(0,0,0), -1)
    cv2.putText(cond,text,org,font,1,(255,255,255),2)

    cv2.imwrite(save_name, cond)
    ####################################################################

    return len(contours) # remove the outerboarder countour
    

class ShanghaiTestBase(Dataset):
    def __init__(self,
                 input_img,
                 size=None,
                 interpolation="bicubic",
                 crop_mode=True
                 ):
        self.crop_mode = crop_mode
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
        if self.crop_mode==True:
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

def ldm_cond_sample(model, batch_size, cv2img, save_name, ddim=True, steps=500):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = next(iter(dataloader))
    seg = x['rgb']
    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)
        save_img(condition, batch_size)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, intermediates = model.sample_log(cond=seg, batch_size=batch_size, ddim=ddim,
                                    ddim_steps=steps, eta=1.)
        samples = model.decode_first_stage(samples)
        # for i in range(0, len(intermediates)):
            # name = "outputs/x_inter"+str(i)+".png"
            # save_image((model.decode_first_stage(intermediates[i])+1)/2.,name)
        
        pred_arr = to_arr((samples+1)/2., batch_size)
        # thr = cv2.equalizeHist(pred_arr)
        # ret, thr = cv2.threshold(thr, 170, 255, cv2.THRESH_TOZERO)
        # clahe = cv2.createCLAHE(40, tileGridSize=(5,5))
        # dst = clahe.apply(pred_arr)
        # dst = cv2.normalize(pred_arr, None, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite('thr.png', pred_arr)
        # print(np.sum(thr))
        crowd_pred = counting(pred_arr,batch_size, cv2img, save_name)
        
        # MAE_list.append(abs((x['crowd'].sum() - crowd_pred)/x['crowd'].sum()))
    return crowd_pred



if __name__ == '__main__':
    seed_everything(23)
    device = torch.device('cuda')
    ################## 옵션 ######### ###################
    ################ 결과 : MSE,  RMSE ##################
    # logname = '2023-06-13T02-30-18_spatial_concat_A_valid'      # 트레이닝 이름 택 1
    # logname = '2023-06-13T09-57-44_spatial_concat'
    # logname = '2023-06-13T02-31-11_spatial_concat_B_novalid'
    # logname = '2023-06-13T02-31-41_spatial_concat_A_novalid'
    logname ='2023-06-18T16-46-49_shanghai_B_crossattn'
    
    checkpoint = 'last.ckpt'                                    # weight 이름 택 1
    # checkpoint = 'epoch=000269.ckpt'                      
    
    #test_mode = 'A'                                             # 테스트 데이터 유형 'A' 또는 'B'
    test_mode = 'B'
    
    crop_mode = False                                            # True: crowd map, rgb 모두 crop
    ddim_True_or_False = False                                   # True: ddim  | False: ddpm
    steps = 800                                                 # ddim: 0~500 | ddpm: 0~800
    ####################################################
    save_name = logname + '.png'                                # <logname>.png 로 시각화 현재 imwrite 비활성화해놈.
    ####### 이 밑으로 코드 고칠 필요 없음. ###############
    timestamp = logname[:19]
    testroot = "/data/Capstone/ShanghaiTech/part_" + test_mode +"/test_data/img/"
    #testroot ="/data/Capstone/valid/valid_data/valid_img_amh/"
    logroot = 'logs/'+ logname
    config_path = logroot + '/configs/' + timestamp+'-project.yaml'
    ckpt_path = logroot + '/checkpoints/' + checkpoint
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, device, None)
    list = os.listdir(testroot)
    list.sort()
    MAE_list = []
    RMSE_list = []
    i = 0
    for imgname in list:
        image_name = testroot+imgname
        gt_name = image_name.replace("IMG", "DENSITY").replace("img", "density")
        # print(image_name, gt_name)

        gtimg =  cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
        if crop_mode == True:
            crop = min(gtimg.shape[0], gtimg.shape[1])
            gtimgh, gtimgw, = gtimg.shape[0], gtimg.shape[1]
            gtimg = gtimg[(gtimgh - crop) // 2:(gtimgh + crop) // 2,
                (gtimgw - crop) // 2:(gtimgw + crop) // 2]

        inputimg = Image.open(image_name)
        cv2image = cv2.cvtColor(np.array(inputimg), cv2.COLOR_RGB2BGR)
        if crop_mode == True:
            crop = min(cv2image.shape[0], cv2image.shape[1])
            h, w, = cv2image.shape[0], cv2image.shape[1]
            img = cv2image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
            cv2image=cv2.resize(img, (512, 512))
        else:
            cv2image=cv2.resize(cv2image, (512, 512))
            
        
        dataset = ShanghaiTest(input_img=inputimg, size=256, crop_mode = crop_mode)#512
        
        crowd_pred = ldm_cond_sample(model, batch_size=1, cv2img = cv2image, save_name=save_name, ddim=ddim_True_or_False, steps=steps)
        #print("num:",i, "img:", imgname, "gt:", np.sum(gtimg)/205, "pred:", crowd_pred, "error rate:", abs(crowd_pred-np.sum(gtimg)/205)/(np.sum(gtimg)/205))
        MAE_list.append(abs(crowd_pred-(np.sum(gtimg)/205))) #MAE
        RMSE_list.append(abs(crowd_pred-(np.sum(gtimg)/205))**2) #MSE
        i+=1
        print(imgname, "gt:", (np.sum(gtimg)/205),"pred:", crowd_pred, "Err", abs(crowd_pred-(np.sum(gtimg)/205)), "SqrErr", abs(crowd_pred-(np.sum(gtimg)/205))**2)
    #print("list(percents):",MAE_list)
    
    print("done.", timestamp, logname, checkpoint, test_mode)
    print("MAE :", sum(MAE_list)/len(list))
    print("RMSE:", math.sqrt(sum(RMSE_list)/len(list)))

    print("MAE_list:", MAE_list)
    print("MSE_list:", RMSE_list)
    # ldm_cond_sample(config_path, ckpt_path, 1, image)
