import glob
import random
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime

import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torch  

from collections import defaultdict

import sys
sys.path.insert(1,"Config")
import configuration as cfg
import parameters as params

#from dataParserHarts import Denoising_parser
#from .augmentation_HARTS import HARTSPolicy
import sys


class Denoising_dataset(Dataset):
    def __init__(self, img_files):
        super(Denoising_dataset,self).__init__()
        self.img_files = img_files
        random.shuffle(self.img_files)

    #center crop 90 X 90
    def imgResize(self,img):
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


    def convert_to_tensor(self, image):
        #image = image[:,:,0]
        image = np.expand_dims(image,axis=2)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))  #CxHXW 
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.type(torch.FloatTensor)
        return image_tensor


    def __getitem__(self,index):
        image_container =  self.img_files[index%len(self.img_files)]
        input_img_path = image_container['input']
        target_img_path = image_container['target']

        img_input = Image.open(input_img_path)
        img_target = Image.open(target_img_path)

        img_input = np.array(img_input)
        img_target = np.array(img_target)

        if img_input.shape[1] > 1024:
            img_input = img_input[:,:1024]
        if img_target.shape[1] > 1024:
            img_target = img_target[:,:1024]

        img_input_tensor = self.convert_to_tensor(img_input)
        img_target_tensor = self.convert_to_tensor(img_target)

        #####augmentation

        return img_input_tensor, img_target_tensor 


    def __len__(self):
        return len(self.img_files)
'''
if __name__ == "__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    parser = Denoising_parser(cfg.data_path, run_started)
    dataset = Denoising_dataset(parser.train_img_file)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("dataset length: ",len(dataset))
    print('dataloader: ',len(dataloader))
    
    for i, (imgs,targets) in enumerate(dataloader):
        print('imgs: ',imgs.size())
        print('targets: ',targets.size())
        exit()
'''