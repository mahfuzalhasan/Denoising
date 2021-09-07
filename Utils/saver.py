import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

import sys
sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/MSGAN/Config")
import configuration as cfg

# tensor to PIL Image
def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    #img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img[:,:,0] *255.0
    return img.astype(np.uint8)

# save an image
def save_img(img,name,path):
    if not os.path.exists(path):
        os.mkdir(path)
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(os.path.join(path, name + '.png'))

# save a set of images
def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Saver():
    def __init__(self, opts, run_id):
        self.display_dir = cfg.logs_dir #os.path.join(opts.display_dir, opts.name)
        self.model_dir = cfg.saved_models_dir
        self.image_dir = cfg.output_path
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.checkpoint

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(os.path.join(self.display_dir, str(run_id)))

    # write losses and images to tensorboard
    # we will save values of losses after every iteration and will show after some iteration
    # will later save after every epoch
    def write_display(self, total_it, it, model):
        #if (total_it + 1) % self.display_freq == 0:
        # write loss
        members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
        for m in members:
            self.writer.add_scalar(m, getattr(model, m), total_it)
            if it % self.display_freq == 0:
                print(m, " : ",getattr(model,m))
        
        # write img
        #image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
        #image_dis = np.transpose(image_dis.detach().numpy(), (1, 2, 0)) * 255
        #image_dis = image_dis.astype('uint8')
        #self.writer.add_image('Image', image_dis, total_it)

    # save result images
    # will write images after some iteration
    def write_img(self, ep, run_id, iteration, model):

        if ep  % self.img_save_freq == 0:
            path = os.path.join(self.image_dir, str(run_id), str(ep))
            if not os.path.exists(path):
                os.makedirs(path)
            assembled_images, real_image, fake_image1, fake_image2 = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (path, iteration)

            real = '%s/gen_real_%05d.jpg' % (path, iteration)
            fake_1 = '%s/gen_fake1_%05d.jpg' % (path, iteration)
            fake_2 = '%s/gen_fake2_%05d.jpg' % (path, iteration)

            #torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=8)
            torchvision.utils.save_image(assembled_images, img_filename, nrow=8)
            torchvision.utils.save_image(real_image, real, nrow=3)
            torchvision.utils.save_image(fake_image1, fake_1, nrow=3)
            torchvision.utils.save_image(fake_image2, fake_2, nrow=3)
        '''
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (path, iteration)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=8)
        '''
    # save model
    # will save model after every epoch
    def write_model(self, ep, run_id, model):
        path = os.path.join(self.model_dir, str(run_id))
        if not os.path.exists(path):
            os.makedirs(path)
        if ep % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (path, ep), ep)

