import os
import torch
from torch.utils.data import DataLoader
from data import train_dataset
from config import *
import unet
from torch import nn
from diffusion import forward

epochs = 200
batch_size = 800

dataloader = DataLoader(
    train_dataset, batch_size, shuffle=True, num_workers=4, persistent_workers=True
)

# 创建模型实例，并移动到指定设备
model = unet.unet(1).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.L1Loss()

model.train()
for epoch in range(epochs):
    epoch_loss = 0  # 用于记录每个epoch的总损失
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(DEVICE) * 2 - 1
        batch_t = torch.randint(0, T, (batch_x.size(0),)).to(DEVICE)

        batch_x_t, batch_noise = forward(batch_x, batch_t)

        batch_predict_t = model(batch_x_t, batch_t)
        loss = loss_fn(batch_predict_t, batch_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加batch损失
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')  # 输出每个epoch的平均损失
    torch.save(model, 'model.pt.tmp')
    os.replace('model.pt.tmp', 'model.pt')
