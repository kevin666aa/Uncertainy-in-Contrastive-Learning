from ast import arg
import enum
import numpy as np
import os
import matplotlib.pyplot as plt
# import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from utils import accuracy
import argparse
import logging
import torch.nn as nn
import sys
from models.resnet_mcdropout import ResNetMCDropout
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2
from PIL import Image



class OODNoLabelDataset(Dataset):
    def __init__(self, order=0):
        self.img_dir, self.img_names, self.ds_name = get_img_dir_name(order)
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(96,96)),
            transforms.ToTensor()
        ])
        # self.crop = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomCrop(size=96),
        #     transforms.ToTensor()
        # ])
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize(image)
        # if image.shape[0] < 96 or image.shape[1] < 96:
        #     image = self.resize(image)
        # else:
        #     image = self.crop(image)
        # print(image.shape)
        return image, -1


base = '/vol/bitbucket/yw2621/datasets/other'
ds_map =  { 0 : 'animal_images',
            1 : 'natural_scenes',
            2 : 'veg_images',
            3 : 'animals10',}

def get_img_dir_name(order=0):
    img_dir = None
    img_names = []

    if order == 0: # animal_images
        img_dir = os.path.join(base, 'animal_images/animal_images')
        img_names = os.listdir(img_dir)[:8000]

    elif order == 1: #'natural_scenes'
        img_dir = os.path.join(base, 'natural_scenes/seg_train/seg_train')
        fds = os.listdir(img_dir)
        offset = int( 8000 / len(fds)) + 1
        for f in fds:
            tmp = os.listdir(os.path.join(img_dir, f))[:offset]
            tmp = [f + '/' + i for i in tmp]
            img_names = img_names + tmp
        img_names = img_names[:8000]

    elif order == 2: # 'veg_images',
        img_dir = os.path.join(base, 'Vegetable Images/train')
        fds = os.listdir(img_dir)
        offset = int(8000 / len(fds)) + 1
        for f in fds:
            tmp = os.listdir(os.path.join(img_dir, f))[:offset]
            tmp = [f + '/' + i for i in tmp]
            img_names = img_names + tmp
        img_names = img_names[:8000]

    elif order == 3: # 'animals10',
        img_dir = os.path.join(base, 'animals10/raw-img')
        fds = os.listdir(img_dir)
        for f in fds:
            tmp = os.listdir(os.path.join(img_dir, f))[:800]
            tmp = [f + '/' + i for i in tmp]
            img_names = img_names + tmp
    else:
        print(f'Unknown Order: {order}, exit.')
        exit(1)

    return img_dir, img_names, ds_map[order]

# batch_size = 256
# shuffle = False
# d0 = OODNoLabelDataset(0)
# d0_loader = DataLoader(d0, batch_size=batch_size,
#                             num_workers=10, drop_last=False, shuffle=shuffle)
# next(iter(d0_loader))
# print('passed')

# d1 = OODNoLabelDataset(1)
# d1_loader = DataLoader(d1, batch_size=batch_size,
#                             num_workers=10, drop_last=False, shuffle=shuffle)
# next(iter(d1_loader))
# print('passed')

# d2 = OODNoLabelDataset(2)
# d2_loader = DataLoader(d2, batch_size=batch_size,
#                             num_workers=10, drop_last=False, shuffle=shuffle)
# next(iter(d2_loader))
# print('passed')

# print(len(d0), len(d1), len(d2))
