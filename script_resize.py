
import cv2
import numpy as np


def imgResize(img):
    h = img.shape[0]
    w = img.shape[1]
    color = (0,0,0)
    new_h = 90
    new_w = 90
    result = np.full((new_h,new_w,3), color, dtype=np.uint8)
    # compute center offset
    xx = (new_w - w) // 2
    yy = (new_h - h) // 2
    result[yy:yy+h, xx:xx+w] = img
    return result

def image_read(img_path):
    img = cv2.imread(img_path)
    img = imgResize(img)
    img_resize = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
    return img_resize

if __name__=="__main__":
    img = image_read("Enter your path")
    