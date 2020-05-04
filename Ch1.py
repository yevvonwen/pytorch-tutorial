# Ch1 Start

import torch
import os
import numpy as np

# select gpu

print(torch.cuda.is_available) # check gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tensor or numpy

np_data = np.ones(6).reshape((2,3))

tensor_data = torch.from_numpy(np_data)
tensor2array = tensor_data.numpy()

ident_matrix = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])

# others API 

torch.zeros(3,3) # 3*3 0 的tensor
temp = torch.rand(3,3) # 3*3 隨機的tensor
temp[0][0].item() # 只取數值 
temp.max() # 取最大值
temp.log2() # 取log2

# https://pytorch.org/docs/stable/torch.html

# to other type

ident_matrix.type()
float_tensor = ident_matrix.to(dtype = torch.float32)
float_tensor.type()

# reshape

flat_tensor = torch.rand(200)
flat_tensor.shape

viewed_tensor = flat_tensor.reshape(1,20,10) 
# viewed_tensor = flat_tensor.view(1,15,15)
viewed_tensor.shape

chw_tensor = viewed_tensor.permute(2, 0, 1) # channel, height, width
# > [1, 20, 10] --> [10, 1, 20]





