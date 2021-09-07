from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

import torch
import torchvision



class HARTSPolicy(object):
    def __init__(self, fillcolor=(0, 0, 0)):
        self.policies = [
            SubPolicy(0.0, "shearY", 5, 0.95, "solarize", 50, fillcolor),
            SubPolicy(0.95, "shearX", 9, 0.0, "translateY", 3, fillcolor),
            SubPolicy(0.9, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.9, "shearX", 6, 0.9, "solarize", 8, fillcolor),
            SubPolicy(0.99, "sharpness", 7, 0.99, "brightness",6, fillcolor),
            SubPolicy(0.0, "sharpness", 5, 0.9, "solarize", 127, fillcolor),
            SubPolicy(0.9, "translateY", 3, 0.1, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "rotate", 9, 0.1, "sharpness", 6, fillcolor),



            '''
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
            '''
        ]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        policy_idx = 1
        return self.policies[policy_idx](img)


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 180, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 255),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=(0,), fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=(0,), fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, -1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])), #color
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),     #color
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),                                #color
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([1, 1])),                                
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, -1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),          #color
            "equalize": lambda img, magnitude: ImageOps.equalize(img),                  #no need gray
            "invert": lambda img, magnitude: ImageOps.invert(img)                       #color
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.operation1_name = operation1
        #magnitude_idx1 = random.randint(4,6)
        self.magnitude1 = ranges[operation1][magnitude_idx1]

        self.p2 = p2
        self.operation2 = func[operation2]
        self.operation2_name = operation2
        #magnitude_idx2 = random.randint(3,6)
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        op1 = ""
        op2 = ""
        if random.random() < self.p1:
            print("op1: ",self.operation1_name)
            img = self.operation1(img, self.magnitude1)
            op1 += self.operation1_name
            print('magnitude1: ',self.magnitude1)
        if random.random() < self.p2:
            print("op2: ",self.operation2_name) 
            #print("mode: ",img.mode)
            print('magnitude2: ',self.magnitude2)
            img = self.operation2(img, self.magnitude2)
            #img2 = img.convert('RGBA')
            #img_converted = self.operation2(img2, self.magnitude2)
            #print("mag2:", self.magnitude2 )
            #fff = Image.new('RGBA', img_converted.size, (128,)*4)
            #out = Image.composite(img_converted, fff, img_converted)
            #print('out mode: ',out.mode)
            #out = out.convert(img.mode)
            #print('out mode conversion: ',out.mode)
            op2 += self.operation2_name
        return img, op1, op2