import torch.nn as nn

import sys
from PIL import Image
import numpy as np
import torch
from utils import get_a_model, device
softmax = nn.Softmax(dim=1)
from utils import enable_dropout
import matplotlib.pyplot as plt

labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
class CamExtractor(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.pretrained = model
        self.layerhook.append(target_layer.register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward_pass(self,x):
        out = self.pretrained(x)
        return self.selected_out, out


# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        enable_dropout(model)
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        #  conv_output shape (imgs, 512, 3, 3),  model_output (imgs, 10)

        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())

        prob = softmax(model_output) # (imgs, 10)
        uct = -torch.sum(prob*torch.log(prob + sys.float_info.min), dim=-1) # (imgs, )
        uct.sum().backward(retain_graph=True)

        # Get weights from gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy() # (imgs, 512, 3, 3)
        weights = np.mean(guided_gradients, axis=(2, 3))  #  (imgs, 512)

        # apply weights to convolutional outputs from target layer
        target = conv_output.data.cpu().numpy() # (imgs, 512, 3, 3)

        weights = np.expand_dims(weights, axis=(2, 3)) # (imgs, 512, 1, 1) expand for broadcast
        cams = np.sum(target * weights, axis = 1) + 1 # (imgs, 3, 3)
        cams = np.maximum(cams, 0) 
                
        def get_min_max_std(array):
            # array = (array - np.min(array)) / (np.max(array) - np.min(array))
            return np.sum(np.std(array, 0)), np.std(np.sum(array,0))
        a, b, c, d = 0, 0, 0, 0
        for i in range(len(input_image)):
            tmpa, tmpb = get_min_max_std(guided_gradients[i])
            print('gradient:', tmpa, tmpb)
            a += tmpa
            b += tmpb

            tmpc, tmpd = get_min_max_std(target[i])
            print('feature map:', tmpc, tmpd)
            c += tmpc
            d += tmpd

            # print(, np.std(np.sum(guided_gradients[i],0)))
            # print(np.sum(np.std(target[i], 0)), np.std(np.sum(target[i], 0)), )
            # a += np.sum(np.std(guided_gradients[i], 0))
            # b += np.std(np.sum(guided_gradients[i],0))
            # c += np.sum(np.std(target[i], 0))
            # d += np.std(np.sum(target[i], 0))

        print(a/len(input_image), b /len(input_image), c/len(input_image), d/len(input_image))
        

        cams_origin_size = np.zeros((1,) + input_image.shape[2:]) 
        for cam in cams:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                        input_image.shape[3]), Image.ANTIALIAS))/255
            cams_origin_size = np.concatenate((cams_origin_size, np.expand_dims(cam, axis=0)), 0)
            
        # cams_origin_size shape: (imgs+1, orginal_im_size_0, orginal_im_size_1)
        return cams_origin_size[1:], prob.data.cpu().numpy(), uct.data.cpu().numpy()


def plot_gradcam(images, predicts, titles, filename= '1', rows=1, cols=3):
    assert cols*rows == len(images), 'Length of images doesn\'t match col*row.'
    assert len(images) == len(titles), 'Length images doesn\'t math length of titles.'
    fig = plt.figure(figsize=(cols*2, rows*2))
    for i, im in enumerate(images):
        fig.add_subplot(rows, cols, i+1)
        plt.axis('off')
        max_ind = np.argmax(predicts[i])
        title_name = titles[i]+ '_' + labels[max_ind] + '_' + str(round(predicts[i][max_ind], 3))
        plt.title(title_name, fontsize=10)
        plt.imshow(im, cmap=plt.cm.hot)

    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/SimCLR/figures/gradcam/' + filename + '.jpg')
    plt.close(fig)

def get_cam(gradcam, X, forward_passes = 10):
    num_imgs = X.shape[0]
    cams = np.zeros((num_imgs, 96, 96))
    probs = np.zeros((num_imgs, 10))
    ucts = np.zeros((num_imgs))
    for i in range(forward_passes):
        tmp_cam, tmp_prob, tmp_uct = gradcam.generate_cam(X) # (num_imgs, 96, 96)
        probs += tmp_prob
        cams += tmp_cam
        ucts += tmp_uct
    return cams/forward_passes, probs/forward_passes, ucts/forward_passes


# class ResNet18(nn.Module):
#     def __init__(self, model):
#         super(ResNet18, self).__init__()
#         # get the pretrained VGG19 network
        
#         # disect the network to access its last convolutional layer

#         # self.features_conv = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, 
#         # model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4)
#         t = [model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, \
#         model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4]
#         self.features_conv =  nn.Sequential(*t)

#         # get the max pool of the features stem
#         self.max_pool = model.backbone.avgpool

#         # get the classifier of the vgg19
#         self.classifier = model.backbone.fc
        
#         # placeholder for the gradients
#         self.gradients = None
    
#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         print('In activations_hook')
#         self.gradients = grad
        
#     def forward(self, x):
#         x = self.features_conv(x)
#         # register the hook
#         h = x.register_hook(self.activations_hook)

#         # apply the remaining pooling
#         x = self.max_pool(x).squeeze()

#         # x = x.view((1, -1))
#         x = self.classifier(x)
#         return x
    
#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients
    
#     # method for the activation exctraction
#     def get_activations(self, x):
#         return self.features_conv(x)




# class GradCamModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.gradients = None
#         self.tensorhook = []
#         self.layerhook = []
#         self.selected_out = None
        
#         #PRETRAINED MODEL
#         self.pretrained = model
#         self.layerhook.append(self.pretrained.backbone.layer4[2].register_forward_hook(self.forward_hook()))
        
#         for p in self.pretrained.parameters():
#             p.requires_grad = True
    
#     def activations_hook(self,grad):
#         print('activations_hook used')
#         self.gradients = grad

#     def get_act_grads(self):
#         return self.gradients

#     def forward_hook(self):
#         def hook(module, inp, out):
#             self.selected_out = out
#             self.tensorhook.append(out.register_hook(self.activations_hook))
#         return hook

#     def forward(self,x):
#         out = self.pretrained(x)
#         return out, self.selected_out