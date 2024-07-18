import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from config import *

pil_to_tensor = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor()
])

tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda x:x*255),
    transforms.Lambda(lambda x:x.type(torch.uint8)),
    transforms.ToPILImage()
])

train_dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=pil_to_tensor)
