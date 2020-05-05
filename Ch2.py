
# Ch2 simple NN

import torch
import torchvision
from torchvision import transforms

train_data_path = ""
valid_data_path = ""
test_data_path = ""

batch_size = 32
epochs = 200

# preprocessing 
# https://pytorch.org/docs/stable/torchvision/transforms.html

trainsforms = transforms.Compose([
      transforms.Resize(),
      transforms.ToTensor(),
      transforms.Normalize(mean = [],
                          std = [])
]) 

train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = transforms)
valid_data = torchvision.datasets.ImageFolder(root = valid_data_path, transform = transforms)
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)


##########################################################################################################################
# creat NN

# 基本上分兩個部分，一個部分寫網路架構，一個部分寫怎麼傳遞

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# other nn.functional : https://pytorch.org/docs/stable/nn.functional.html

class SimpleNet(nn.Module):
  
  def __init__(self):
    
    super(Net, self).__init__()
    self.fc1 = nn.Linear(512, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)
    
  def forward(self):
    # x = x.view(-1,512) # photo
    x = F.relu(self.fc1)
    x = F.relu(self.fc2)
    x = F.mse_loss(self.fc3)
    
    return x
 
simplenet = SimpleNet()

# optimizer
# https://pytorch.org/docs/stable/optim.html

optimizer = optim.Adam(simplenet.parameter(), lr = 0.001)

# start training
# 要做哪件事就開哪個模式

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
          
          
train(simplenet, optimizer, torch.nn.mseloss, train_data_loader, valid_data_loader)
torch.save(simplenet, "root/")          
    
    
    
    








