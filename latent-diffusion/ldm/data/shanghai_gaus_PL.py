import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ShanghaiBase(Dataset):
    def __init__(self,
                 data_dir,
                 size=None,
                 interpolation="bicubic",
                 ):
        # self.data_paths = txt_file
        self.data_dir = data_dir
        self.image_paths = []
        self.cond_paths = []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files: #"DENSITY_0201.png"
                if file.endswith(".png"):
                    file_path = os.path.join(subdir, file) #"train/train_data/train_density" "/" "DENSITY_0201.png"
                    self.image_paths.append(file_path)
                    self.cond_paths.append(file_path.replace("DENSITY", "IMG").replace("density", "img"))
                    
        # with open(self.data_paths, "r") as f:
        #     self.image_paths = f.read().splitlines()



        self._length = len(self.image_paths)
        self. labels = {
            "image_path_": [l for l in self.image_paths],
            "cond_path_": [l for l in self.cond_paths],
        }
        #print('labels;---------', self.labels)
        # self. labels = {
        #     "relative_file_path_": [l for l in self.image_paths],
        #     "file_path_": [os.path.join(self.data_dir, l)
        #                    for l in self.image_paths],
        # }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        # self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        # self.flip = TF.hflip()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # example = dict()
        example = dict((k, self.labels[k][i]) for k in self.labels)
        density = Image.open(example["image_path_"])

        #width, height = density.size
        #print(width,height)
        #pixel_sum = 0
        #for y in range(height):
            #for x in range(width):
                #pixel_value = density.getpixel((x, y))
                #pixel_sum += pixel_value
        #print("lllllllllll",pixel_sum,"lllllllllll")
        cond = Image.open(example["cond_path_"])
        # rgb_path = self.image_paths[i]
        # cond_path = rgb_path[i].replace("IMG", "DENSITY").replace("img", "density")
    
        # example = dict((k, self.labels[k][i]) for k in self.labels)
        # image = Image.open(example["file_path_"])
        for idx, image in enumerate([density, cond]):
            if not image.mode == "RGB":
                image = image.convert("RGB")
                #pixel_array = np.array(image)
                #pixel_sum = np.sum(pixel_array)
                #print("kkkkkkkkkkkkkkk",pixel_sum,"kkkkkkkkkkkkkkkk")

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8) #우선 crop제외
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

            image = Image.fromarray(img)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)

            # if probility == 1: # 증강
            #     # image = self.flip(image)
            #     image = TF.hflip(image)
            image = np.array(image).astype(np.uint8)
            #image = (image / 127.5 - 1.0).astype(np.float32)<-normalize 2번 적용
            if idx == 0:
                key = "density"
                GTcount=int(image.sum()/30000)
                print("-----------GT:",GTcount,"-----------")
                example[key] =(image / 127.5 - 1.0).astype(np.float32)
                print("jjjjjjjjjjjjjjjjjj",np.shape(example["density"]),np.sum(example["density"]),"jjjjjjjjjjjjjjjjj")
                #print("zzzzzzzzzzzzzzzzz",np.sum(image),"zzzzzzzzzzzzzzz")
                #image=image/3 #rgb로 바뀌면서 값이 3배가 된다.#0829
                #print("wwwwwwwwwwwwwwwwwww",np.sum(image),"wwwwwwwwwwwwwwwww") #이 값은 어쨋든 density map을 만들때 값을 곱한 값의 합이다. ex)30000 그리고 crop된 것이라서 숫자가 작을 수 밖에 없다.
            else: 
                key = "rgb"
                example[key] =(image / 127.5 - 1.0).astype(np.float32) #image는 (256x256x3)이다.따라서 counting으로 따지면 127.5을 나누고 -196608을 빼는 것과 같다. 다시 255로 돌리려면 x.sum()[0~255까지의 픽셀합]=((y[-1~1까지의 픽셀합] + 196608)x127.5
            #image에 0이 있다 따라서 성립 안된디ㅏ!!!!! 직접 ddpm2에 Count GT값 불러야 한다. 코드에서는 그냥 해당 x_start의 합과 예측치를 비교하여 loss를 만들어봄, 잘 안나오면 다시 시도해보기
            #example[key] =(image + 1).astype(np.float32)  
            #if(key=="density"):
            #    print("jjjjjjjjjjjjjjjjjj",np.shape(example["density"]),np.sum(example["density"]),"jjjjjjjjjjjjjjjjj")
            #else:
            #    print("iiiiiiiiiiiiiiiiiii",np.sum(example["rgb"]),"iiiiiiiiiiiiiiiiiiii")
        # example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        # return example
        #print("tttttttttttttttttt",example,"tttttttttttttttttttt")
        return example


class ShanghaiTrain(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="../train/train_data/train_density_gaus_PL2",**kwargs)


class ShanghaiValidation(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="../valid/valid_data/valid_density_gaus_PL2",**kwargs)
        