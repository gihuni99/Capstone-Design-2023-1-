import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ShanghaiTestBase(Dataset):
    def __init__(self,
                 data_dir,
                 size=None,
                 interpolation="bicubic"
                 ):
        # self.data_paths = txt_file
        self.data_dir = data_dir
        self.cond_paths = []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files: #"DENSITY_0201.png"
                if file.endswith(".png"):
                    file_path = os.path.join(subdir, file) #"train/train_data/train_density" "/" "DENSITY_0201.png"
                    self.cond_paths.append(file_path)
                    
        # with open(self.data_paths, "r") as f:
        #     self.image_paths = f.read().splitlines()



        self._length = len(self.cond_paths)
        self. labels = {
            "cond_path_": [l for l in self.cond_paths],
        }


        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # example = dict()
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["cond_path_"])

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
    


class ShanghaiBase(Dataset):
    def __init__(self,
                 data_dir,
                 size=None,
                 interpolation="bicubic"
                #  flip_p=0.5
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
        cond = Image.open(example["cond_path_"])
        probility = np.random.randint(2)
        # rgb_path = self.image_paths[i]
        # cond_path = rgb_path[i].replace("IMG", "DENSITY").replace("img", "density")
    
        # example = dict((k, self.labels[k][i]) for k in self.labels)
        # image = Image.open(example["file_path_"])
        for idx, image in enumerate([density, cond]):
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

            if probility == 1: # 증강
                # image = self.flip(image)
                image = TF.hflip(image)
            image = np.array(image).astype(np.uint8)
            #image = (image / 127.5 - 1.0).astype(np.float32)<-normalize 2번 적용
            if idx == 0:
                key = "density"
            else: key = "rgb"
            example[key] = (image / 127.5 - 1.0).astype(np.float32)
        # example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        # return example
        #print("tttttttttttttttttt",example,"tttttttttttttttttttt")
        return example


class ShanghaiTrain(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="../train/train_data/train_density10000",**kwargs)


class ShanghaiValidation(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="../valid/valid_data/valid_density10000",**kwargs)
        
        
class ShanghaiTest(ShanghaiTestBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="../test/test_data/test_2imgs",**kwargs)