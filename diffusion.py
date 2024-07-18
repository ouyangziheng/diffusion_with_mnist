import torch
from config import *
from data import train_dataset

batas = torch.linspace(0.0001, 0.02, T).to(DEVICE)
alphas = 1 - batas

# 累乘(前向传播)
alphas_cum = torch.cumprod(alphas, -1).to(DEVICE)

# 累乘补1 对应的alpha_t-1 时刻的累乘法
alphas_cum_prev = torch.cat([torch.tensor([1.0]).to(DEVICE), alphas_cum[:-1]], dim=-1).to(DEVICE)

# denoise 的方差  在torch库中会按照元素操作
variance = ((1 - alphas) * (1 - alphas_cum) / (1 - alphas_cum_prev)).to(DEVICE)


# batch_x 为batch个初始图像[batchsize, channel, width, height] 
# batch_t 为batch中的某一个图片的步数

def forward(batch_x, batch_t):
    # 生成和batch_x 一样的张量
    gauss_noise_t = torch.randn_like(batch_x)
    # 对每一个像素添加对应的噪声
    alpha_batch_cum = (alphas_cum_prev[batch_t].view(batch_x.size()[0], 1, 1, 1)).to(DEVICE)

    batch_x_t = torch.sqrt(alpha_batch_cum) * batch_x +  torch.sqrt(
        1 - alpha_batch_cum
    ) * gauss_noise_t
    return batch_x_t, gauss_noise_t
