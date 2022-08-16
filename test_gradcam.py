
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import get_a_model, device
from inference import  get_aug_loader, load_three_models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import sys
import torch.nn as nn
softmax = nn.Softmax(dim=0)
from mygrad_cam import GradCam, get_cam, plot_gradcam
from ood import OODNoLabelDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


print('Start test_gradcam')
model0, model1, model2 = load_three_models('resnet18', 0.1)   
gradcam0 = GradCam(model0, model0.backbone.layer4[2])
gradcam1 = GradCam(model1, model1.backbone.layer4[-1])
gradcam2 = GradCam(model2, model2.backbone.layer4[2])

num_imgs = 20
forward_passes = 20

# test_loader = get_aug_loader('no aug', shuffle=True)
# X, y = next(iter(test_loader))
# X = X[:num_imgs, : ,:, :].to(device) # X shape: (num_imgs, 3, 96, 96)
# cams0, l0, uct0 = get_cam(gradcam0, X, forward_passes)
# cams1, l1, uct1 = get_cam(gradcam1, X, forward_passes)
# cams2, l2, uct2 = get_cam(gradcam2, X, forward_passes)

# for i in range(num_imgs):
#     single_X = np.moveaxis(X[i].cpu().numpy().squeeze(), 0, 2)
#     v0, _ = show_cam_on_image(single_X, cams0[i], use_rgb=False)
#     v1, _ = show_cam_on_image(single_X, cams1[i], use_rgb=False)
#     v2, _ = show_cam_on_image(single_X, cams2[i], use_rgb=False)
#     plot_gradcam([v0, v1, v2], [l0[i], l1[i], l2[i]], 'z_gradcam_' + str(i))


for o in range(3):
    dataset = OODNoLabelDataset(o)
    dataloader = DataLoader(dataset, batch_size=256 * 2,
                        num_workers=10, drop_last=False, shuffle=True)

    num_imgs = 20
    X, _ = next(iter(dataloader))
    X = X[:num_imgs, : ,:, :].to(device) # X shape: (num_imgs, 3, 96, 96)

    cams0, l0, uct0 = get_cam(gradcam0, X, forward_passes)
    cams1, l1, uct1 = get_cam(gradcam1, X, forward_passes)
    cams2, l2, uct2 = get_cam(gradcam2, X, forward_passes)

    for i in range(num_imgs):
        single_X = np.moveaxis(X[i].cpu().numpy().squeeze(), 0, 2)
        v0, _ = show_cam_on_image(single_X, cams0[i], use_rgb=False)
        v1, _ = show_cam_on_image(single_X, cams1[i], use_rgb=False)
        v2, _ = show_cam_on_image(single_X, cams2[i], use_rgb=False)
        plot_gradcam([v0, v1, v2], [l0[i], l1[i], l2[i]], 'z_ood_gradcam_' + str(i + o*10))


print('End test_gradcam')