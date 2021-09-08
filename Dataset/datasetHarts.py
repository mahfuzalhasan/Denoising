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
    def __init__(self, df, transform=None):
        super(Denoising_dataset,self).__init__()
        self.df = df
        self.transform = transform
        df.sample(frac=1)
        self.img_input = self.df['input'].tolist()
        self.img_target = self.df['target'].tolist()

    def convert_to_tensor(self, image):
        #image = image[:,:,0]
        image = np.expand_dims(image,axis=2)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1)) #CxHXW 
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.type(torch.FloatTensor)
        return image_tensor


    def __getitem__(self,index):

        img_input = Image.open(self.img_input[index])
        img_target = Image.open(self.img_target[index])

        img_input = np.array(img_input)
        img_target = np.array(img_target)

        if img_input.shape[1] > 1024:
            img_input = img_input[:,:1024]
        if img_target.shape[1] > 1024:
            img_target = img_target[:,:1024]

        if self.transform:
            augmented = self.transform(image=img_input, mask=img_target)
            img_input = augmented['image']
            img_target = augmented['mask']
        img_input_tensor = self.convert_to_tensor(img_input)
        img_target_tensor = self.convert_to_tensor(img_target)

        #####augmentation

        return img_input_tensor, img_target_tensor 


    def __len__(self):
        return len(self.img_input)
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