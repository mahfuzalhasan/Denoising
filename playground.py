import torch
#create tensor with random data, and multiply with a scalar
b = torch.ones((2,3,2,2))
b = b.cuda(0)
print(b)
rand_tensor = (0.8)*b#torch.rand(b.size())
rand_tensor = rand_tensor.cuda(0)
#print tensor
print('rand: ',rand_tensor)
one_hot = torch.full((b.size()), 0.8).cuda(0)

print('one hot: ',one_hot)