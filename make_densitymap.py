import numpy as np
import math
import os
import cv2
from PIL import Image
import scipy.io as sio

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

def create_dmap(img, gtLocation, sigma, downscale=1.0):
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

            kernel = GaussianKernel((25, 25), sigma=sigma)

            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    maxx = np.max(densityMap)
    return densityMap, maxx

def load_point(gt_mat):
    loc = sio.loadmat(gt_mat)
    # print(loc['image_info'][0, 0][0, 0][0])
    loc = loc['image_info'][0, 0][0, 0][0].astype(np.float32)
    return loc
    

if __name__ == '__main__':
        imgdir = os.listdir("/data/Capstone/ShanghaiTech_val/part_B/valid_data/img")
        
        for i in range(0, len(imgdir)):
            img = "/data/Capstone/ShanghaiTech_val/part_B/valid_data/img/"+imgdir[i]
            gt = "GT_"+imgdir[i].replace("png", "mat")
            gt_mat = "/data/Capstone/ShanghaiTech_val/part_B/train_data/ground-truth/"+gt
            # print(img, depth_matfile, gt_mat)
            img2 = cv2.imread(img)
            height, width, cns = img2.shape
            raw_width, raw_height = width, height

            loc = load_point(gt_mat)
            
            dmap, maxx = create_dmap(img2, loc, 0.5, downscale=2.0)

            dmap = 205*dmap
            imgname = "/data/Capstone/ShanghaiTech_val/part_B/valid_data/img"+imgdir[i].replace("jpg", "png")
            name = "//data/Capstone/ShanghaiTech_val/part_B/valid_data/density/"+imgdir[i].replace("IMG", "DENSITY").replace("jpg", "png")
            # cv2.imwrite(imgname, img2)
            cv2.imwrite(name, dmap)
        print("done")