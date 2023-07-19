import numpy as np
import math
import os
import cv2
from PIL import Image
import scipy.io as sio
from scipy import spatial
from scipy.spatial import KDTree
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
import glob
import h5py
import time
from joblib import Parallel, delayed
import math

def generate_density_map(img_path):
    print(img_path)
    mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
    mat = scipy.io.loadmat(mat_path)
    imgfile = image.load_img(img_path)
    img = image.img_to_array(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    output_path = img_path.replace(__DATASET_ROOT, __OUTPUT_NAME).replace('.jpg', '.h5').replace('images','ground-truth-h5')
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("output", output_path)
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = k
    print(k,k.min(),k.max())
    k=(k/k.max())*255.0
    print(k,k.min(),k.max())
    # .h5 파일에서 이미지 데이터 읽어오기
    with h5py.File(output_path, 'r') as h5_file:#추가
    # 이미지 데이터는 'image' 등의 적절한 키로 저장되어 있다고 가정
        image_data = h5_file['density'][:]#추가
        
    img=Image.fromarray(image_data)#추가
    img = img.convert('RGB')#추가
    img.save("fffff.png")#추가
    return img_path

def GaussianKernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian kernel which is equal to MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    radius_x, radius_y = [(radius-1.)/2. for radius in shape]
    y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
    h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumofh = h.sum()
    if sumofh != 0:
        h /= sumofh
    return h


# def create_dmap(img, gtLocation, depth, beta=0.25, downscale=8.0):
#     width, height = img.size
#     raw_width, raw_height = width, height
#     width = math.floor(width / downscale)
#     height = math.floor(height / downscale)
#     raw_loc = gtLocation
#     gtLocation = gtLocation / downscale
#     gaussRange = 25
#     # kernel = GaussianKernel(shape=(25, 25), sigma=3)
#     pad = int((gaussRange - 1) / 2)
#     densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
#     for gtidx in range(gtLocation.shape[0]):
#         if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
#             xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
#             yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
#             x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
#             x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)
#             y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
#             y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
#             depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
#             kernel = GaussianKernel((25, 25), sigma=beta * 5 / depth_mean)
#             densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
#     densityMap = densityMap[pad:pad + height, pad:pad + width]
#     return densityMap

def cv_dmap(img, gtLocation, depth, sigma, downscale=1.0):
    height, width, cns = img.shape
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale

    densityMap = np.zeros((int(height), int(width)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]))
            yloc = int(math.floor(gtLocation[gtidx, 1]))
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)

            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            # if(y_up>int(raw_height)/2 and depth_mean <)
            # print(np.mean(arr))
            # depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            arr = depth[y_down:y_up, x_down:x_up]
            depth_mean = np.min(arr)
            # print(depth_mean)
            # print(depth_mean)
            if depth_mean != 0:
                densityMap = cv2.circle(densityMap, (xloc, yloc), int(3*sigma/depth_mean), (255,255,255), -1)
            else:
                densityMap = cv2.circle(densityMap, (xloc, yloc), int(3*sigma), (255,255,255), -1)
    return densityMap

def create_dmap(img, gtLocation, depth, sigma, downscale=1.0):
    height, width, cns = img.shape
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale
    # gaussRange = 25
    gaussRange = 25
    pad = int((gaussRange - 1) / 2)
    densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
            yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)

            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            # depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            arr = depth[y_down:y_up, x_down:x_up]
            depth_mean = np.min(arr)
            # print(np.sum(depth[y_down:y_up]), np.sum(depth[x_down:x_up]))
            # print(depth_mean)
            
            if depth_mean != 0:
                # kernel = GaussianKernel((25, 25), sigma=sigma/ depth_mean)
                kernel = GaussianKernel((25, 25), sigma=sigma)
                #kernel = GaussianKernel((25, 25), sigma=10*sigma / depth_mean)
            else:
                #kernel = GaussianKernel((25, 25), sigma=10*sigma)
                kernel = GaussianKernel((25, 25), sigma=sigma)
            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    maxx = np.max(densityMap)
    return densityMap, maxx


def load_depth(depth_matfile):
    depth = sio.loadmat(depth_matfile)
    depth = depth['depth']
    depth = depth / depth.max()
    # print(depth)
    # depth = depth * 255
    # depth = depth / 255
    # depth = depth.astype(np.uint8)
    # print(depth)
    return depth

def load_point(gt_mat):
    loc = sio.loadmat(gt_mat)
    loc = loc['point'].astype(np.float32)
    return loc
    

if __name__ == '__main__':
        imgdir = os.listdir("ShanghaiTech/part_A/test_data/images")
        maxx_arr = []
        for i in range(0, len(imgdir)):
            img = "ShanghaiTech/part_A/test_data/images/"+imgdir[i]
            mat_path = imgdir[i].replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
            #depth = imgdir[i].replace("IMG", "DEPTH").replace("png", "mat")
            #gt = imgdir[i].replace("IMG", "GT").replace("png", "mat")
            depth_matfile = "ShanghaiTech/part_A/test_data/ground-truth/"+mat_path
            #gt_mat = "valid/valid_data/valid_gt/"+gt
            # print(img, depth_matfile, gt_mat)
            img2 = cv2.imread(img)
            height, width, cns = img2.shape
            raw_width, raw_height = width, height
            # load annotation
            # annot is a numpy array (N, 5).
            # N means there are N objects in the image, 5 means {x1,y1,x2,y2,class_id}

            depth = load_depth(depth_matfile)
            loc = load_point(depth_matfile)
            # dmap = cv_dmap(img2, loc, depth, 1.2, downscale=2.0)
            
            dmap, maxx = create_dmap(img2, loc, depth, 0.5, downscale=2.0)
            maxx_arr.append(maxx)
            ## 0~1
            # print(maxx)
            # print(imgdir[i], np.sum(dmap)) # 사람수
            # dmap = 255 *dmap / maxx
            dmap = 205*dmap
            #일단 보류
            #dmap3=dmap+10e-13
            #print(dmap3.sum())
            #print(dmap,dmap.max(),dmap.sum())
            #dmap=255.*(dmap/dmap.max()-dmap.min()) #stretching
            # dmap = 30000*dmap
            # print(dmap.max(),dmap.min(), dmap.sum())
            
            #dmap2=(dmap/255.)
            
            #print("9999",dmap2.sum())
            #print("d",dmap.sum()/10000)
            
            
            
            # ret,thresh = cv2.threshold(dmap,10,255,0)
            # dmap = dmap.astype(np.uint8)
            # dmap = np.dstack([dmap, dmap, dmap])
            # data = Image.fromarray(dmap)
            name = "mkdata/"+imgdir[i].replace("IMG", "DENSITY")
            # dmap = cv2.resize(dmap, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite(name, dmap)
            # data.save(name)
            # print(i, "번째")
        # print(np.mean(maxx_arr))
        print("done")