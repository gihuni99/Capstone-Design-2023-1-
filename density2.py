import cv2
import numpy as np

def crop(image):
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
    (w - crop) // 2:(w + crop) // 2]
    return img

if __name__ == '__main__':
    multiplied = 205
    img2 = cv2.imread('/data/Capstone/ShanghaiTech/part_A/train_data/density2/DENSITY_1.png', cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread('/data/Capstone/test/test_data/test_2imgs/IMG_0012.png_zsample.png', cv2.IMREAD_GRAYSCALE)
    #img3 = cv2.imread('/data/Capstone/test/test_data/test_1imgs/IMG_0206.png_sample.png', cv2.IMREAD_GRAYSCALE)
    #img3=cv2.imread('/data/Capstone/DENSITY_0001.png', cv2.IMREAD_GRAYSCALE)
    #img2=crop(img2)
    #img3=crop(img3)
    # height, width = img2.shape
    # raw_width, raw_height = width, height
    # print(img2.shape)
    # img2 = cv2.resize(img2, (256,256))
    img2 = img2/multiplied
    # img3 = img3/multiplied
    # img3 = img3
    #cv2.imwrite('GT_crop', img2)
    #cv2.imwrite('sample_crop', img3)
    print("GT:", np.sum(img2))
    # print("Sample: ", np.sum(img3))
    # print("Error: ", np.sum(img2)-np.sum(img3))