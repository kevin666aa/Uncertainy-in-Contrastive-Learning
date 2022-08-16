
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
from mygrad_cam import GradCam

class TotalUncertaintyTarget:
    def __init__(self, category=0):
        self.category = category

    def __call__(self, model_output):
        prob = softmax(model_output)
        return -torch.sum(prob*torch.log(prob + sys.float_info.min), dim=-1)

def plot_gradcam(images, name= '1', titles = ['Dropout', 'CL', 'Dropout+CL'], rows=1, cols=3):
    assert cols*rows == len(images), 'Length of images doesn\'t match col*row.'
    assert len(images) == len(titles), 'Length images doesn\'t math length of titles.'

    fig = plt.figure(figsize=(cols*2, rows*2))
    for i, im in enumerate(images):
        fig.add_subplot(rows, cols, i+1)
        plt.axis('off')
        plt.title(titles[i])
        plt.imshow(im, cmap=plt.cm.hot)

    plt.tight_layout()
    plt.savefig('z/' + name + '.jpg')
    plt.close(fig)

def get_cam(gradcam, targets, X, forward_passes = 10):
    num_imgs = X.shape[0]
    cams = np.zeros((num_imgs, 96, 96))
    for i in range(forward_passes):
        cams += gradcam(input_tensor=X, targets=targets) # (num_imgs, 96, 96)

    cams /= forward_passes
    return cams
#, aug_smooth=True, eigen_smooth=True

# dropout_model0 = get_a_model(False, 'resnet18', 0.1, 'mcdropout/checkpoint_best.pth.tar')
model0, model1, model2 = load_three_models('resnet18', 0.1)
gradcam0 = GradCAM(model=model0, target_layers=[model0.backbone.layer4[2]], use_cuda= device=='cuda')
gradcam1 = GradCAM(model=model1, target_layers=[model1.backbone.layer4[-1]], use_cuda= device=='cuda')
gradcam2 = GradCAM(model=model2, target_layers=[model2.backbone.layer4[2]], use_cuda= device=='cuda')
num_imgs = 10

targets = None
targets = [TotalUncertaintyTarget()] * num_imgs
test_loader = get_aug_loader('no aug')
X, _ = next(iter(test_loader))
X = X[:num_imgs, : ,:, :].to(device)

cams0 = get_cam(gradcam0, targets, X, forward_passes = 10)
cams1 = get_cam(gradcam1, targets, X, forward_passes = 10)
cams2 = get_cam(gradcam2, targets, X, forward_passes = 10)

for i in range(num_imgs):
    single_X = np.moveaxis(X[i].cpu().numpy().squeeze(), 0, 2)
    v0, _ = show_cam_on_image(single_X, cams0[i], use_rgb=True)
    v1, _ = show_cam_on_image(single_X, cams1[i], use_rgb=True)
    v2, _ = show_cam_on_image(single_X, cams2[i], use_rgb=True)
    plot_gradcam([v0, v1, v2], '1_gradcam_' + str(i))

# # print(grayscale_cam.shape)
# # # In this example gayscale_cam has only one image in the batch:
# for i in range(num_imgs):
#     t = np.moveaxis(X.cpu().numpy()[i], 0, 2)
#     visualization, heatmap = show_cam_on_image(t, grayscale_cam[i, :], use_rgb=False)
#     cv2.imwrite(f"z/test_{i}_uct_onelayer.jpg", visualization)



# plt.imshow(cam, cmap=plt.cm.hot)
# plt.savefig('heatmap.jpg')
# out, selected_out = mymodel(X)
# print(out.shape, selected_out.shape)

# one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_().to(device)
# one_hot_output[0][3] = 1

# out.backward(gradient=one_hot_output, retain_graph=True)

# gradients = mymodel.get_act_grads().detach().cpu()
# pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# # weight the channels by corresponding gradients
# for i in range(512):
#     selected_out[:, i, :, :] *= pooled_gradients[i]
    
# # average the channels of the activations
# heatmap = torch.mean(selected_out, dim=1).squeeze().detach().cpu()

# # relu on top of the heatmap
# # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
# heatmap = np.maximum(heatmap, 0)
# print(heatmap)
# normalize the heatmap





# class TotalUncertaintyTarget:
#     def __init__(self, category=0):
#         self.category = category

#     def __call__(self, model_output):
#         prob = softmax(model_output)
#         return -torch.sum(prob*torch.log(prob + sys.float_info.min), dim=-1)

# dropout_model0 = get_a_model(False, 'resnet18', 0.1, 'mcdropout/checkpoint_best.pth.tar')
# cam = GradCAM(model=dropout_model0, target_layers=[dropout_model0.backbone.layer4[2].conv2], use_cuda= device=='cuda')
# num_imgs = 10

# targets = None
# targets = [TotalUncertaintyTarget()] * num_imgs
# test_loader = get_aug_loader('no aug')
# X, _ = next(iter(test_loader))
# X = X[:num_imgs, : ,:, :].to(device)

# grayscale_cam = cam(input_tensor=X, targets=targets, aug_smooth=True, eigen_smooth=True)
# print(grayscale_cam.shape)
# # In this example gayscale_cam has only one image in the batch:
# for i in range(num_imgs):
#     t = np.moveaxis(X.cpu().numpy()[i], 0, 2)
#     visualization, heatmap = show_cam_on_image(t, grayscale_cam[i, :], use_rgb=False)
#     cv2.imwrite(f"z/test_{i}_uct_onelayer.jpg", visualization)
# plt.imshow(heatmap, cmap=plt.cm.hot)

# plt.savefig('heatmap.jpg')



# from torchvision.models import resnet50

# model = resnet50(pretrained=False)
# target_layers = [model.layer4[-1]]
# print(model.layer4[-1])


# dataset = OODNoLabelDataset(2)
# dataloader = DataLoader(dataset, batch_size=5,
#                         num_workers=10, drop_last=False, shuffle=False)
# num_imgs = 10
# X, _ = next(iter(dataloader))
# print('aaa')
# # X = X[1, : ,:, :].to(device) # X shape: (num_imgs, 3, 96, 96)
# for i in range(5):
#     tmp = np.moveaxis(X[i].cpu().numpy().squeeze(), 0, 2)
#     plt.imshow(tmp)
#     # plt.savefig(f"{i}.jpg")
#     cv2.imwrite(f"{i}.jpg", np.uint8(tmp*255))
