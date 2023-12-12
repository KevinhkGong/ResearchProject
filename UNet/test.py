import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from PIL import Image
import numpy as np

dir_img = Path('./data/test.py_img/')
dir_mask = Path('./data/test.py_mask/')

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    """
    image = Image.open('./data/test.py_mask/0cdf5b5d0ce1_01_mask.gif')
    image_np = np.array(image)
    print(image_np.shape)
    plt.imshow(image.resize((2144,1424)))
    print("start")
    plt.show()
    """

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #print(dataset[0].keys())
    #print(dataset[0]['image'].shape)
    #print(dataset[0]['mask'].shape)
    #plt.imshow(dataset[0]['image'].permute(1,2,0))
    #plt.imshow(dataset[0]['mask'])
    #plt.show()
    # 2. Split into train / validation partitions
    n_train = len(dataset)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = dataset
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    images, labels = next(iter(train_loader))





model = UNet(n_channels=3, n_classes=2, bilinear=False)
model = model.to(memory_format=torch.channels_last)
train_model(
            model=model,
            epochs=5,
            batch_size=1,
            learning_rate=1e-5,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            img_scale=0.5,
            val_percent=10.0 / 100,
            amp=False
        )


