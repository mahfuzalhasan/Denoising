import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import pickle
import random
import cv2
import pickle
import statistics

import sys
sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/MSGAN/Exp_1/Config")

import xml.etree.ElementTree as ET
import json

import configuration as cfg
import parameters as params



class Denoising_parser(object):
    def __init__(self, data_path, run_id):
        super(Denoising_parser,self).__init__()
        self.data_path = data_path  
        self.train_img_file = []
        self.valid_img_file = []
        self.img_files = []

        data_split_dir = os.path.join(cfg.data_split,run_id)
        if not os.path.exists(data_split_dir):
            os.makedirs(data_split_dir)

        train_image_path = os.path.join(data_split_dir,"train_images.pkl")
        validation_image_path = os.path.join(data_split_dir, "valid_images.pkl")

        if os.path.exists(train_image_path):
            print('load from pre-saved distribution')
            self.train_img_file = pickle.load(open(train_image_path, 'rb'))
            #self.valid_img_file = pickle.load(open(validation_image_path, 'rb'))
        else:       #only this will get called now
            print("Creating train and validation")
            self.train_img_file = self._load_SEM_data(self.data_path)
            #self.valid_img_file = self._load_harts_data(self.data_path, valid=True)
            pickle.dump(self.train_img_file, open(train_image_path, 'wb'))
            #pickle.dump(self.valid_img_file, open(validation_image_path, 'wb'))

        random.shuffle(self.train_img_file)
        #random.shuffle(self.valid_img_file)

    def __len__(self):
        return len(self.train_img_file)

    def _split(self):
        random.shuffle(self.img_files)

        index = int(0.8 * len(self.img_files))
        train_img_file = self.img_files[0:index+1]
        valid_img_file = self.img_files[index+1:]
        return train_img_file, valid_img_file  

             
    def _load_SEM_data(self, img_dir, valid=False):
        img_files = []
        if not valid:
            print("Train")
        else:
            print("Val")
        for img_folder in os.listdir(img_dir):
            store = {}
            for img_name in os.listdir(os.path.join(img_dir,img_folder)):
                if '6' in img_name:
                    continue
                store['set'] = int(img_folder)
                store['input'] = os.path.join(img_dir,img_folder,img_name)
                target_name = img_name.replace(img_name[1],'6')
                store['target'] = os.path.join(img_dir,img_folder,target_name)
                img_files.append(store)
        return img_files
    

if __name__=="__main__":
    parser = Denoising_parser(cfg.data_path, run_id="5")
    ##print(parser.classes_to_idx)
    #print(len(parser.img_files))
    ##print(parser.img_files[100])