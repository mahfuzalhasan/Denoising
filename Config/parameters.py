#training
num_epoch = 1000
batch_size = 8
#resume = None #"/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/saved_models/02-10-21_1341_msgan/00074.pth"
resume = None #"/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/saved_models/02-11-21_0752_msgan/00180.pth"
gpu = 0
smoothing_value = 0.85

#test
result_dir = "/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/test_output"
name = "02-11-21_0752_msgan"
num = 100


#optimizer
learning_rate = 1e-4

#data
num_classes = 6
channels = 3
img_size = 64
nz = 100#latent_dim = 100
img_height = 64
img_width = 64
img_save_freq = 3
num_channel = 1



#model related
checkpoint = 1      #no of epoch to save model


#dataset
validation_set = "DT4_Mag12"

#output
display_freq = 150