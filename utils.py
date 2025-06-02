from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

import medmnist
from medmnist import INFO, Evaluator


# Taken from the original notebook given: TP2-students.ipynb
# Added normalization
def load_medMNIST_data() -> tuple:
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    data_flag = 'bloodmnist'
    download = True
    BATCH_SIZE = 128

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing -> normalization to [-1,1] range
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load all splits
    train = DataClass(split='train', transform=data_transform, download=True)
    val = DataClass(split='val', transform=data_transform, download=True)
    test = DataClass(split='test', transform=data_transform, download=True)

    # Combine them into a single dataset
    full_dataset = ConcatDataset([train, val, test])
    dataloader = DataLoader(full_dataset, batch_size=128, shuffle=True)

    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # assess image size
    print(f"Image size: {images[0].shape}")

    # assess label size
    print(f"Label size: {labels[0].shape}")

    # assess label values
    s = set()
    for i in range(len(labels)):
        s.add(labels[i].item())
    print(f"Label values: {s}")

    return dataloader, full_dataset
