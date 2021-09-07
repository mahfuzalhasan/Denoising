import argparse
import os
import numpy as np
import math
from datetime import datetime
import time
import sys
import copy

from tensorboardX import SummaryWriter


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch  
from torch.optim import Adam, lr_scheduler


from Dataset.dataParserHarts import Denoising_parser
from Dataset.datasetHarts import Denoising_dataset
from Model.model import CDCGAN
from Utils.saver import Saver, AvgMeter
import Config.configuration as cfg
import Config.parameters as params

from unet import UNet





# ----------
#  Training
# ----------

def psnr(input, target):
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def train(run_id, use_cuda, epoch, dataloader, model, criterion, optimizer, writer):

    total_losses = []
    start_time = time.time()
    model.train()
    psnr_meter = AvgMeter()

    output_dir = os.path.join(cfg.output_path,run_id,str(epoch))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (imgs,targets) in enumerate(dataloader):
        '''
        if imgs.size(0) != params.batch_size:
            continue
        '''
        if use_cuda:
            imgs, targets = imgs.to(f'cuda:{model.device_ids[0]}'), targets.to(f'cuda:{model.device_ids[0]}')
            criterion.to(f'cuda:{model.device_ids[0]}')

        imgs_denoised = model(imgs)
        loss = criterion(imgs_denoised, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())

        for j in range(params.batch_size):
            imgs_denoised = imgs_denoised.cpu()
            targets = targets.cpu()
            psnr_meter.update(psnr(imgs_denoised[j], targets[j]).item())

        if i%3 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [AVRG PSNR: %f]"
                % (epoch, params.num_epoch, i, len(dataloader), np.mean(total_losses), psnr_meter.avg)
            )
            save_image(imgs.data[:6], os.path.join(output_dir,"batch_"+str(i)+".jpg"), nrow=3, normalize=True)
            save_image(imgs_denoised.data[:6], os.path.join(output_dir,"batch_"+str(i)+"_denoised.jpg"), nrow=3, normalize=True)
            save_image(targets.data[:6], os.path.join(output_dir,"batch_"+str(i)+"_targets.jpg"), nrow=3, normalize=True)

    time_taken = time.time() - start_time
    print('time taken: ',time_taken)


    if epoch % params.checkpoint == 0:
        save_file_path = os.path.join(cfg.saved_models_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr':psnr_meter.avg

        }
        torch.save(states, save_file_path)
    

    print(f'Training Epoch {epoch}::: Total Loss:{np.mean(total_losses)} PSNR:{psnr_meter.avg} time taken:{time_taken}')

    sys.stdout.flush()

    writer.add_scalar('Training Total Loss', np.mean(total_losses), epoch)
    writer.add_scalar('Training PSNR', np.mean(psnr_meter.avg), epoch)

    return model, optimizer


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_reconstruction(run_id, use_cuda):
    # Initialize generator and discriminator
    writer = SummaryWriter(os.path.join(cfg.logs_dir, str(run_id)))
    model = UNet(in_channels=1, out_channels=1)
    if use_cuda:
        model = nn.DataParallel(model, device_ids = [0, 2, 3])
        model.to(f'cuda:{model.device_ids[0]}')
    ep0 = -1
    if params.resume is not None:
        ep0 = model.resume(params.resume)
        print('resume from epoch: ',ep0)
    ep0 += 1

    # Dataset preparation
    data_parser = Denoising_parser(cfg.data_path, run_id) 
    img_files = copy.deepcopy(data_parser.train_img_file)
    print('file size: ',len(img_files))
    
    saver = Saver(params, run_id)

    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=params.learning_rate, betas=(0.9,0.99), eps=1e-8)
    print('done')
    
    for epoch in range(ep0, params.num_epoch):
        train_dataset = Denoising_dataset(img_files)
        dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        print("len dataset: ",len(train_dataset))
        print("len dataloader: ",len(dataloader))
        model, optimizer = train(run_id, use_cuda, epoch, dataloader, model, criterion, optimizer, writer)

if __name__ == "__main__":

    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    run_started =str(run_started)+'_msgan'
    use_cuda = torch.cuda.is_available()
    print('use cuda: ',use_cuda)

    train_reconstruction(run_started, use_cuda)
