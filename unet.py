import torch
from torch import nn
from time_position import TimePositionEmbedding
from config import *

# Convblock with time embedding
class convblock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emd_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.time_emb_linear = nn.Linear(time_emd_size, out_channel)

        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x, t_emb):
        x = self.seq1(x)
        t_emb = self.relu(self.time_emb_linear(t_emb).view(x.size(0), x.size(1), 1, 1))
        return self.seq2(x + t_emb)

# U-Net architecture
class unet(nn.Module):
    def __init__(self, img_channel, channels=[64, 128, 256, 512, 1024]):
        super().__init__()

        channels = [img_channel] + channels

        self.time_emd = nn.Sequential(
            TimePositionEmbedding(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )

        self.encoder_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_convs.append(
                convblock(channels[i], channels[i + 1], embedding_size)
            )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder_convs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.decoder_convs.append(
                convblock(channels[-1 - i], channels[-2 - i], embedding_size)
            )

        self.up_cov = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.up_cov.append(
                nn.ConvTranspose2d(
                    channels[-1 - i], channels[-2 - i], kernel_size=2, stride=2
                )
            )

        self.output = nn.Conv2d(
            channels[1], img_channel, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, t):
        t_emb = self.time_emd(t)

        residual = []
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, t_emb)
            if i != len(self.encoder_convs) - 1:
                residual.append(x)
                x = self.maxpool(x)

        for i, upconv in enumerate(self.up_cov):
            x = upconv(x)
            res_x = residual.pop(-1)
            x = torch.cat((res_x, x), dim=1)
            x = self.decoder_convs[i](x, t_emb)

        return self.output(x)
