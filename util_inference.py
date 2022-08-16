import torch
import os
import sys
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd

import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image, make_grid

from data_aug.gaussian_blur import GaussianBlur
from data_aug.add_noise import GaussianNoise, SaltAndPepperNoise
from data_aug.cutout import Cutout
from utils import accuracy, get_a_model


model_names = {
    0 : 'dropout',
    1 : 'dropout+aug',
    2 : 'CL',
    3 : 'dropout+CL'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

augs = {
     'gaussian_blur': np.arange(0.001, 10, 0.2), # 0 - 2 
    #  'crop_resize': np.arange(1, 0, -0.07), # 0 - 0.92  # 7
     'gaussian_noise': np.arange(0, 0.2, 0.018),
     'color_jitter': np.arange(1, 6, 0.5), # 1 - 2 # 3
    #  'salt_and_peper_noise': np.arange(0, 0.1, 0.009),
    #  'cutout': np.arange(0, 100, 9), 
    }



def load_four_models(arch, dropout):
    m0 = get_a_model(False, arch, dropout, 'mcdrop_no_aug/checkpoint_best.pth.tar')
    m1 = get_a_model(False, arch, dropout, 'mcdrop_aug/checkpoint_best.pth.tar')
    m2 = get_a_model(True, arch, dropout, 'clnodrop/resnet18_lr4_bs256_origin/finetuned.pth.tar')
    m3 = get_a_model(False, arch, dropout, 'resnet18_lr4_dropout0.1/finetuned.pth.tar')
    print('Four models loaded.')
    return [m0, m1, m2, m3]

def get_aug_loader(aug_name, i=0, download=True, shuffle=False, batch_size=256, return_loader=True):
    if aug_name == 'gaussian_noise':
        trans = transforms.Compose([
            transforms.ToTensor(),
            GaussianNoise(0., i)
        ])
    elif aug_name == "color_jitter":
        ran = (i, i + 0.01)
        color_jitter = transforms.ColorJitter(ran, ran, ran, i/10-0.1)
        trans = transforms.Compose([
                transforms.RandomApply([color_jitter], p=1), 
                transforms.ToTensor()])
    elif aug_name == "salt_and_peper_noise":
        trans = transforms.Compose([
            transforms.ToTensor(),
            SaltAndPepperNoise(amount=i)])
    elif aug_name == "crop_resize":
        trans = transforms.Compose([
            transforms.RandomResizedCrop(96, (i,i+0.001)),
            transforms.ToTensor()])
    elif aug_name == 'gaussian_blur':
        trans = transforms.Compose([
            GaussianBlur(int(0.1 * 96), i),
            transforms.ToTensor(),])
    elif aug_name == 'cutout':
        trans = transforms.Compose([
            Cutout(1, i),
            transforms.ToTensor()])
    else:
        print('Aug not in aug_names. Use orginal Dataset.')
        trans = transforms.ToTensor()

    if not return_loader:
        return trans

    test_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='test', download=download,
                                  transform=trans)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=12, drop_last=False, shuffle=shuffle)
    return test_loader


def get_monte_carlo_predictions(data_name,
                                data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
    total_acc = 0
    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    labels = torch.Tensor([]) # for calculating acc

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        for j, (image, label) in enumerate(data_loader):
            if i == 0:
                labels = torch.cat((labels, label))
            label = label.to(device)
            image = image.to(device)
            with torch.no_grad():
                output = model(image)
                # total_acc += accuracy(output, label, sum=True)[0]
                output = softmax(output) # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))
        
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)
    
    # Accuracy #
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    total_acc = accuracy(torch.Tensor(mean).to(device), labels.to(device), sum=False)[0].item() # scalar

    # TOTAL uncertainty/entropy #
    total_entropy = -np.sum(mean*np.log(mean + sys.float_info.min), axis=-1) # shape (n_samples,)
    
    # DATA uncertainty/entropy #
    tmp = -np.sum(dropout_predictions*np.log(dropout_predictions + sys.float_info.min), axis=-1) # shape (forward_passes, n_samples)
    data_entropy = np.mean(tmp, axis=0) # shape (n_samples,)
    
    # MODEL uncertainty/entropy #
    model_uncertain = np.mean(total_entropy) - np.mean(data_entropy)

    # format latex
    s = '{} {:.3f}& {:.3f}& {:.3f}& {:.3f}\\\\'.format(data_name,
                                                    total_acc, 
                                                    np.mean(total_entropy), 
                                                    np.mean(data_entropy),
                                                    model_uncertain)
    # print(data_name, np.mean(total_entropy), np.var(total_entropy))
    print(s)
    return total_entropy, data_entropy, total_acc, s



# def plot_correlation_2x2(plots, x, aug_name):
#     ''' Plot correlation between uncertainty/acc vs aug.
#         the ylimits is hard coded.
#         the draw_area is to specify that in what range the CL models have seen the data aug intensity. also hard coded.
#     Param:
#         plots: in shape (4, num_models, -). 
#             4 corresponds to different metric specified in  'names' variable. 
#             num_models correspond to number of models.
#             - is the length of the variations.
#     '''
#     if x[0] > x[1]:
#         print('Identify cropresize')
#         x = 1-x
#     x = x[:10]

#     names = ['Accuracy', 'Total Uncertainty', 'Data Uncertainty' , 'Model Uncertainty']
#     ylimits = [(20, 85), (0.6, 1.8), (0.6, 1.8), (0, 0.43)]
#     ylabel = ['accuracy'] + ['uncertainty'] * 3
#     columns, rows = 2, 2
#     draw_area = {'color_jitter': (1, 2),
#                 'crop_resize': (0, 0.62),
#                 'gaussian_blur': (0.1, 2)}
#     fig = plt.figure(figsize=(columns*6, rows*6))

#     for i, data in enumerate(plots):
#         fig.add_subplot(rows, columns, i+1)
#         if aug_name in draw_area.keys():
#             y = np.arange(0, 100, 0.1)
#             plt.fill_betweenx(y, draw_area[aug_name][0], draw_area[aug_name][1], facecolor = 'lightgray')

#         for model_index in range(len(data)):
#             plt.plot(data[model_index], label=model_names[model_index], linewidth=4)

#         plt.title(names[i])
#         if i == 3:
#             plt.legend(prop={'size': 18})
#         plt.ylim(ylimits[i])

#     plt.tight_layout()    
#     plt.savefig('/vol/bitbucket/yw2621/SimCLR/figures/aug_curve/Uncertainty vs ' + aug_name + '.jpg')
#     plt.close(fig)

def plot_correlation_2x2(plots, x, aug_name):
    ''' Plot correlation between uncertainty/acc vs aug.
        the ylimits is hard coded.
        the draw_area is to specify that in what range the CL models have seen the data aug intensity. also hard coded.
    Param:
        plots: in shape (4, num_models, -). 
            4 corresponds to different metric specified in  'names' variable. 
            num_models correspond to number of models.
            - is the length of the variations.
    '''
    if x[0] > x[1]:
        print('Identify cropresize')
        x = 1-x
    x = x[:10]

    names = ['Accuracy', 'Total Uncertainty', 'Data Uncertainty' , 'Model Uncertainty']
    ylimits = [(20, 85), (0.6, 1.8), (0.6, 1.8), (0, 0.43)]
    ylabel = ['accuracy'] + ['uncertainty'] * 3
    columns, rows = 2, 2
    draw_area = {'color_jitter': (0, 3),
                'crop_resize': (0, 8),
                'gaussian_blur': (0, 9)}
    fig = plt.figure(figsize=(columns*6, rows*6))

    for i, data in enumerate(plots):
        fig  = plt.figure()
        if aug_name in draw_area.keys():
            y = np.arange(0, 100, 0.1)
            plt.fill_betweenx(y, draw_area[aug_name][0], draw_area[aug_name][1], facecolor = 'lightgray')

        for model_index in range(len(data)):
            plt.plot(data[model_index], label=model_names[model_index], linewidth=4)

        # plt.title(names[i])
        if i == 3:
            plt.legend(prop={'size': 18})
        # plt.xlabel('intensity', fontsize = 20)
        # plt.ylabel(ylabel[i], fontsize = 20)
        plt.ylim(ylimits[i])
        plt.xticks(fontsize= 15)
        plt.yticks(fontsize= 15)
        plt.subplots_adjust(top = 0.97, bottom = 0.08, right = 0.98, left = 0.1, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/single_aug_curve/{aug_name}_{i}.pdf')
        plt.close(fig)

def plot_correlation_1x1(data, x, title='Total Uncertainty', name='plt.jpg', xlabel = 'Noise', ylabel='Uncertainty'):
    ''' Plot correlation between aug vs one uncertainty. Single Plot
    '''
    plt.figure()
    plt.title(title)
    for model_index in range(len(data)):
        plt.plot(x, data[model_index], label=model_names[model_index])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('/vol/bitbucket/yw2621/SimCLR/figures/aug_curve', name))

def plot_uncertainty_vs_one_aug(aug_name, models, forward_passes):
    if aug_name not in augs.keys():
        print(f"{aug_name} not found. Choose valid data augmentation.")
        exit(1)

    num_intensity = 10
    size = (len(models),  num_intensity)
    total_uncertainties, data_uncertainties, model_uncertainties = np.zeros(size), np.zeros(size),np.zeros(size)
    accuracies = np.zeros(size)

    def run_a_model(model_name, model_index, model, i, test_loader):
        t, d, acc, _= get_monte_carlo_predictions(model_name,
                                    test_loader, 
                                    forward_passes=forward_passes, 
                                    model=model,  # !
                                    n_classes=10, 
                                    n_samples=len(test_loader.dataset))
        t, d = np.mean(t), np.mean(d)
        total_uncertainties[model_index][i] = t
        data_uncertainties[model_index][i] = d
        model_uncertainties[model_index][i] = t - d
        accuracies[model_index][i] = acc
    
    intensities = augs[aug_name]
    for i in range(num_intensity):
        test_loader = get_aug_loader(aug_name, intensities[i])
        print(aug_name, i)
        for model_index, model in enumerate(models):
            run_a_model(model_names[model_index], model_index, model, i, test_loader)
        del test_loader
        print('\n')

    list_in = [accuracies, total_uncertainties, data_uncertainties, model_uncertainties]
    plot_correlation_2x2(list_in, intensities[:num_intensity], aug_name)    


def plot_roc_3x3(oods, in_ds, y_true, ds_name):
    '''
    oods shape: (3, num_models=4, num_images),  3 means three uncertainties, num_images = (len(oods dataset) + len(ind dataset))
    in_ds shape: same as oods

    y_true shape: (len(oods dataset) + len(ind dataset), )
    '''
    num_models = len(in_ds[0])
    rows, columns = 3, 3
    fig = plt.figure(figsize=(24, 24))
    titles = ['Total', 'Data', 'Model']
    for i in range(3): # 3 plots Total', 'Data', 'Model']
        ax = fig.add_subplot(rows, columns, i+1)   
        for model_index in range(num_models):
            y_pred = np.concatenate((in_ds[i][model_index], oods[i][model_index]))
            y_pred = (y_pred - min(y_pred))/ (max(y_pred) - min(y_pred))
        
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            if i == 0:
                plt.plot(fpr, tpr, label = model_names[model_index] + ' AUC = %0.2f' % roc_auc, linewidth=4)
            else:
                plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc, linewidth=4)
        
        # plt.title('ROC ' + titles[i])
        plt.legend(loc = 'lower right', prop={'size': 25})
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
    
    for i in range(3): # 3 plots Total', 'Data', 'Model']
        fig.add_subplot(rows, columns, i+4)  
        # plt.title("OOD density - "  + titles[i])
        for model_index in range(num_models):
            sns.kdeplot(oods[i][model_index], fill = True, label=model_names[model_index], bw_adjust=.8, common_norm=False)
        if i == 2:
            plt.legend(loc = 'upper right', prop={'size': 32})
        # plt.xlim((0,))

    for i in range(3): # 3 plots Total', 'Data', 'Model']
        fig.add_subplot(rows, columns, i+7)  
        plt.title("In Distribution - "  + titles[i])
        for model_index in range(num_models):
            sns.kdeplot(in_ds[i][model_index], fill = True, label=model_names[model_index], bw_adjust=.8, common_norm=False)
        if i == 2:
            plt.legend(loc = 'upper right', prop={'size': 32})

    plt.tight_layout()
    plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/roc/ROC_{ds_name}.jpg')
    print(f'Saving /vol/bitbucket/yw2621/SimCLR/figures/roc/ROC_{ds_name}.jpg')
    plt.close(fig)



def plot_single_roc(oods, in_ds, y_true, ds_name):
    '''
    oods shape: (3, num_models=4, num_images),  3 means three uncertainties, num_images = (len(oods dataset) + len(ind dataset))
    in_ds shape: same as oods

    y_true shape: (len(oods dataset) + len(ind dataset), )
    '''
    num_models = len(in_ds[0])
    titles = ['Total', 'Data', 'Model']
    for i in range(3): # 3 plots Total', 'Data', 'Model']
        fig  = plt.figure() 
        for model_index in range(num_models):
            y_pred = np.concatenate((in_ds[i][model_index], oods[i][model_index]))
            y_pred = (y_pred - min(y_pred))/ (max(y_pred) - min(y_pred))
        
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            if i == 0:
                plt.plot(fpr, tpr, label = model_names[model_index] + ' AUC = %0.2f' % roc_auc, linewidth=4)
            else:
                plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc, linewidth=4)

        plt.legend(loc = 'lower right', prop={'size': 20})
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.subplots_adjust(top = 0.97, bottom = 0.08, right = 0.97, left = 0.07, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/single_roc/ROC_{ds_name}_{i}.pdf')
        # print(f'Saving /vol/bitbucket/yw2621/SimCLR/figures/single_roc/ROC_{ds_name}.jpg')
        plt.close(fig)
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
    
    print(len(oods[0][0]))
    for i in range(3): # 3 plots Total', 'Data', 'Model']
        fig  = plt.figure() 
        for model_index in range(num_models):
            ax = sns.kdeplot(oods[i][model_index], fill = True, label=model_names[model_index], bw_adjust=0.8, common_norm=True)
        if i == 2:
            plt.legend(loc = 'upper right', prop={'size': 20})
        plt.subplots_adjust(top = 0.97, bottom = 0.08, right = 0.98, left = 0.02, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.set(ylabel=None)
        plt.xticks(fontsize= 15)
        plt.yticks([])
        # plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/single_roc/density_{ds_name}_{i}.pdf')
        # print(f'Saving /vol/bitbucket/yw2621/SimCLR/figures/single_roc/density_{ds_name}_{i}.jpg')
        plt.close(fig)

    for i in range(3): # 3 plots Total', 'Data', 'Model']
        fig  = plt.figure()
        for model_index in range(num_models):
            ax = sns.kdeplot(in_ds[i][model_index], fill = True, label=model_names[model_index], bw_adjust=.8, common_norm=True)
        if i == 2:
            plt.legend(loc = 'upper right', prop={'size': 20})
        ax.set(ylabel=None)
        plt.xticks(fontsize= 15)
        plt.yticks([])
        plt.subplots_adjust(top = 0.97, bottom = 0.08, right = 0.98, left = 0.02, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/single_roc/density_stl10_{i}.pdf')
        # print(f'Saving /vol/bitbucket/yw2621/SimCLR/figures/single_roc/density_stl10_{i}.jpg')
        plt.close(fig)
