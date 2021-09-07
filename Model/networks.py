import  torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/MSGAN/Exp_1/Config")

import configuration as cfg 
import parameters as params
from Model import model

####################################################################
#------------------------- Generator -------------------------------
####################################################################
class generator(nn.Module):
    # initializers
    def __init__(self, opts, d=64):
        super(generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(opts.nz+ opts.num_classes, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn =nn.BatchNorm2d(d*2)
        self.relu3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn =nn.BatchNorm2d(d)
        self.relu4 = nn.ReLU()
        self.deconv5 = nn.ConvTranspose2d(d, opts.num_channel, 4, 2, 1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, input, label):
        print('input G size: ',input.size())
        x = torch.cat([input.unsqueeze(2).unsqueeze(3), label], 1)
        print('input to network: ',x.size())
        x = self.relu1(self.deconv1_bn(self.deconv1(x)))
        print('d1: ',x.size())
        x = self.relu2(self.deconv2_bn(self.deconv2(x)))
        print('d2: ',x.size())
        x = self.relu3(self.deconv3_bn(self.deconv3(x)))
        print('d3: ',x.size())
        x = self.relu4(self.deconv4_bn(self.deconv4(x)))
        print('d4: ',x.size())
        x = self.tanh(self.deconv5(x))
        print('d5: ',x.size())
        return x

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class discriminator(nn.Module):
    # initializers
    def __init__(self, opts, d=64):
        super(discriminator, self).__init__()
        self.conv1= nn.Conv2d((opts.num_channel + opts.num_classes), d, 4, 2, 1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.sigmoid = nn.Sigmoid()


    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, input, label):
        #print('label size: ',label.size())
        label = label.expand(label.shape[0], label.shape[1], input.shape[2], input.shape[3])
        #print('extended label size: ',label.size())
        x = torch.cat([input, label], 1)
        #print('input d:',x.size())
        x = self.lrelu1(self.conv1(x))
        #print('c1: ',x.size())
        x = self.lrelu2(self.conv2_bn(self.conv2(x)))
        #print('c2: ',x.size())
        x = self.lrelu3(self.conv3_bn(self.conv3(x)))
        #print('c3: ',x.size())
        x = self.lrelu4(self.conv4_bn(self.conv4(x)))
        #print('c4: ',x.size())
        #x = self.lrelu5(self.projection_bn(self.projection(x)))
        ##print('projection: ',x.size())
        x = self.sigmoid(self.conv5(x))
        #print('c5: ',x.size())
        return x

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


if __name__=='__main__':
    model = model.CDCGAN(params)
    cuda = True if torch.cuda.is_available() else False
    #print("cuda: ",cuda)
    x = torch.randn(16,3,64,64)
    labels = torch.from_numpy(np.random.randint(0,5,size=(16,1)))
    print(labels.shape)
    labels = labels.type(torch.LongTensor)
    if cuda:
        model.setgpu(params.gpu)
        #print(cuda)
        x = x.cuda(params.gpu)
        labels = labels.cuda(params.gpu)
    dis = discriminator(params)
    out = dis(x,labels)

    #model.update_D(x,labels)
    #model.update_G()

    

    
