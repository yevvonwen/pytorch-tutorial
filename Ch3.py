# ch3  CNN

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# simple CNN

n_classes = 10

class CNN(nn.Module):
  
  def __init__(nn.Module):
    super(CNN, self).__init__()
    self.features = nn.Sequential(
       nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
       nn.ReLU(),
       nn.Maxpool2d(kernel_size = 3, stride = 2),
       nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
       nn.ReLU(),
       nn.Maxpool2d(kernel_size = 3, stride = 2),       
       nn.Conv2d(192, 256, kernel_size = 3, padding = 1),
       nn.ReLU(),
       nn.Maxpool2d(kernel_size = 3, stride = 2),
       nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
       nn.ReLU(),
       nn.Maxpool2d(kernel_size = 3, stride = 2),       
    )
    self.avgpool = nn.AdaptiveAvegPool2d((6,6))
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256*6*6, 4096)
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Linear(4096, n_classes)
    )  
 
  def forward(self, x):
    x = self.feature(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
  

# pytorch_hub
# load pretrained model
# https://pytorch.org/docs/stable/hub.html

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

# 寫def train 迴圈開train




  
      
