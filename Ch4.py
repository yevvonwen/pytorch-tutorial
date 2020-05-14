import torch
import torchvision
from torchvision import models, transforms
import torch.optim as optimizer

# transfer learning and other tricks

transfer_model = models.ResNet50(pretrained = True)

# if you want to fix layers

for name, param in transfer_model.named_parameters():
    param.requres_grad = False
    
# if you do not want to fix batch normalize

# for name, param in transfer_model.named_parameters():
#     if("bn" not in name):
#         param.requres_grad = False
    
# change final classification layers

transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 500),
                                 nn.ReLu(),
                                 nn.Dropout(),
                                 nn.Linear(500, 2)
)

# in_features 抓 variable

# grid search!! 

import math
import matplotlib as plt

def find_lr(model, loss_fn, optimizer, init_value = 1e-8, final_value = 10):
    
    num_in_epoch = len(train_loader)-1
    update_step = (final_value / init_value) ** (1 / num_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    
    for batch in train_loader:
        batch_num = batch_num + 1
        inputs, labels = data
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        
        output = model(inputs)
        loss = loss_fn(output, labels)
        
        # no loss explodes
        
        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]
        
        # record best loss
        
        if loss < best_loss or batch_num == 1:
            best_loss = loss
            
        # store the values
        
        losses.append(loss)
        log_lrs.append(math.log10(lr))
        
        # backward
        
        loss.backward()
        optimizer.step()
        
        # updata lr 
        
        lr = lr * update_step
        optimizer.param_groups[0]["lr"] = lr
        
return log_lrs[10:-5], losses[10:-5] 

# main()

logs, losses = find_lr() # model, loss_fn, optimizer ++
plt.plot(logs, losses)

# 每層設不同的 lr

# 主要是用在TF learning上，因為會選擇凍結前面的層數，但有時候又希望可以稍微訓練一下後面的層數，加上最後的FC，所要調整的層度不同，因此設不同的lr 或許會比較合適。
# 直接訓練一整個NN就用同一個

found_lr = 1e-2

optimizer = optimizer.Adam([
    {'params': transfer_model.layer4.parameters(), 'lr': found_lr / 3},
    {'params': transfer_model.layer3.parameters(), 'lr': found_lr / 9},
    ], lr = found_lr)

# 開那幾層可以train

unfreeze_layers = [transfer_model.layer4,  transfer_model.layer3]

for layer in unfreeze_layers:
    for param in layer.parameters():
        param.requres_grad = True
        
# data augementation

# 在 transforms 裡做 in Ch2

trainsforms = torchvision.transforms.Compose([
    transforms.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = 0), # random change ... value from 0-1
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),   # random rotate default p = 0.1
    transforms.RandomRotation(degree = 45)
    
    transforms.Resize(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [],
                          std = []),
        
])

# other data aug
# https://pytorch.org/docs/stable/torchvision/transforms.html

from PIL import Image as im

# transform color space

def transform_color_space(x):
    output = x.im.convert("HSV")
    return output

# all
color_transform = transforms.Lambda(lambda x: transform_color_space)

# Random
random_color_transform = transforms.RandomApply([color_transform]) # default p = 0.5

# ensemble model
predictions = [m[i].fit(input) for i in models]
avg_prediction = torch.stack(b).mean(0).argmax() # stack tensor的一個維度的元素相疊加
