import torch

img_size = 48
T = 1000 #steps
DEVICE = "cuda" if torch.cuda.is_available() == True else "cpu"

count_nums = 10
embedding_size = 8

LORA_ALPHA=1    # lora
LORA_R=8    # lora rank
