import torch
import torchvision
from torchvision import models
import torch.optim as optimizer

import math
import matplotlib as plt

import argparse

# set param

parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--batch', type=int, default=32)

parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--class-num', type = int, default = 1000,)

args = parser.parse_args()

train_data_path = "/home/jovyan/colorful-moth/data/model/20200514/train.csv"
valid_data_path = "/home/jovyan/colorful-moth/data/model/20200514/valid.csv"
test_data_path = "/home/jovyan/colorful-moth/data/model/20200514/test.csv"

train_trainsforms = torchvision.transforms.Compose([
    #torchvision.transforms.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = 0), # random change ... value from 0-1
    #torchvision.transforms.RandomHorizontalFlip(p = 0.5),
    #torchvision.transforms.RandomVerticalFlip(p = 0.5),   # random rotate default p = 0.1
    torchvision.transforms.RandomRotation(degree = 45, p = 0.5)
    
    transforms.Resize(),
    transforms.ToTensor(),    
])

test_trainsforms = torchvision.transforms.Compose([
    #torchvision.transforms.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = 0), # random change ... value from 0-1
    #torchvision.transforms.RandomHorizontalFlip(p = 0.5),
    #torchvision.transforms.RandomVerticalFlip(p = 0.5),   # random rotate default p = 0.1
    #torchvision.transforms.RandomRotation(degree = 45, p = 0.5)
    
    transforms.Resize(),
    transforms.ToTensor(),    
])

train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = train_transforms)
valid_data = torchvision.datasets.ImageFolder(root = valid_data_path, transform = train_transforms)
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = train_trainsforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch)
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size = args.batch)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch)

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

for name, param in model.named_parameters():
    param.requres_grad = True

model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 500),
                                 nn.ReLu(),
                                 nn.Dropout(),
                                 nn.Linear(500, 2))

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs= 200):
    for epoch in range(epochs):
    
    training_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, target = batch
      
        inputs = inputs.cuda()
        target = target.cuda()
        
        output = model(inputs)
        loss = loss_fn(output, target)
      
        loss.backward()
      
        optimizer.step()
      
        train_loss = train_loss+ loss.data.item()
      
        training_loss = training_loss/len(train_iterator)
    
        model.eval()
        mse_correct = 0

        for batch in val_loader:
            inputs, target = batch
            inputs = inputs.cuda()
      
            outputs = model(input)
            loss = loss_fn(outputs, targets)
      
            valid_loss = valid_loss+ loss.data.item()
      
            valid_loss = valid_loss/len(valid_iterator)
    
        print('Epoch: {}, Training_loss: {:.2f}', 
        Valid_loss: {:.2f}'.format(epoch, training_loss, valid_loss))
          
          

train(model, optimizer, torch.nn.mseloss, train_data_loader, valid_data_loader)
torch.save(model, "/home/jovyan/colorful-moth/data/model/20200514/model.h5")          
    



