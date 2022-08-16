from ast import arg

import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image, make_grid

from ood import OODNoLabelDataset
from utils import accuracy, check_load_ckpt, enable_dropout, get_a_model
from saliency import pgd_saliency_all_uncertainty, pgd_saliency_ce_loss, pgd_saliency_all_models
from util_inference import get_aug_loader, load_four_models, get_monte_carlo_predictions, plot_uncertainty_vs_one_aug, plot_roc_3x3
from util_inference import labels, augs, model_names, plot_single_roc
from pytorch_grad_cam.utils.image import show_cam_on_image
from mygrad_cam import GradCam, get_cam, plot_gradcam, CamExtractor
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

# for data aug
# plot roc, plot gradcam

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

    
def main_show_aug_effects(num_imgs=10):
    test_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='test', download=True,
                                    transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256,
                                num_workers=12, drop_last=False, shuffle=False)
    tensor_to_pil = transforms.ToPILImage()
    tensors, _ = next(iter(test_loader))
    # tensors = tensors[:num_imgs,:,:,:]

    
    # for aug_name in augs:
    #     t = augs[aug_name]
    #     samples = torch.empty((1,3,96,96))
    #     for j in range(num_imgs):
    #         for i in range(10):
    #             trans = get_aug_loader(aug_name, t[i], return_loader=False)
    #             ddd = trans(tensor_to_pil(tensors[j])).unsqueeze(0)
    #             samples = torch.cat((samples, ddd), 0)
    #     samples = samples[1:]
    #     print(samples.shape)
    #     samples = make_grid(samples, nrow=10, padding=4, normalize=False,
    #                         range=None, scale_each=False, pad_value=0)

    #     plt.figure(figsize = (20,15))
    #     plt.axis('off')
    #     show(samples)
    #     save_image(samples, 'aug_img/' + aug_name + '.jpg')
    #     print('Saving', aug_name + '.jpg')

    tensors = tensors[0,:,:,:]
    samples = torch.empty((1,3,96,96))
    # samples = torch.cat((samples, tensors.unsqueeze(0)), 0)
    for i in range(10):
        for aug_name in augs:
            trans = get_aug_loader(aug_name, augs[aug_name][i], return_loader=False)
            ddd = trans(tensor_to_pil(tensors)).unsqueeze(0)
            samples = torch.cat((samples, ddd), 0)

    samples = samples[1:]
    print(samples.shape)
    samples = make_grid(samples, nrow=6, padding=4, normalize=False,
                            range=None, scale_each=False, pad_value=0)

    # plt.figure(figsize = (20,15))
    plt.axis('off')
    show(samples)
    save_image(samples, 'figures/aug_sample.pdf')


def init_gradcam_models(MODELS):
    # initialize model and grad cam extractor
    num_models = len(MODELS)
    gradcam_models = []
    for model_index in range(num_models):
        if model_index != 2:
            gradcam_models.append(GradCam(MODELS[model_index], MODELS[model_index].backbone.layer4[2]))
        else:
            gradcam_models.append(GradCam(MODELS[model_index], MODELS[model_index].backbone.layer4[-1]))
    
    return gradcam_models

def get_gradcam_outputs(gradcam_models, X, forward_passes):
    cams, predicts, ucts = [], [], []
    for model_index in range(len(gradcam_models)):
        print('Model ', model_index)
        tmp_cam, tmp_l, tmp_uct = get_cam(gradcam_models[model_index], X.to(device), forward_passes)
        cams.append(tmp_cam)
        predicts.append(tmp_l)
        ucts.append(tmp_uct)
        print('\n\n\n')
    return cams, predicts, ucts


def main_visualize_ood_gradcam(MODELS, num_imgs=10, forward_passes=20):
    gradcam_models = init_gradcam_models(MODELS)

    for o in range(4):
        if o != 3:
            continue
        if o == 3:
            dataloader = get_aug_loader('no aug')
            ds_name = 'stl10'
        else:
            dataset = OODNoLabelDataset(o)
            dataloader = DataLoader(dataset, batch_size=256 * 2,
                                num_workers=10, drop_last=False, shuffle=True)
            ds_name = dataloader.dataset.ds_name

        X, _ = next(iter(dataloader))
        X = X[:num_imgs, : ,:, :].to(device)

        cams, predicts, ucts = get_gradcam_outputs(gradcam_models, X, forward_passes)
        
        offsets = num_imgs//10
        for i in range(offsets):
            samples = X[i*10:(i+1)*10, : ,:, :].to(device)
            cams, predicts, ucts = get_gradcam_outputs(gradcam_models, samples, forward_passes)
            
            samples = np.moveaxis(samples.cpu().numpy().squeeze(), 1, 3)

            # for model_index in range(len(gradcam_models)):
            #     for img_index in range(10):
            #         tmp_v, _ = show_cam_on_image(samples[img_index], cams[model_index][img_index], use_rgb=False)
            #         # plt.imshow(tmp_v)
            #         # plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/gradcam/test.jpg')
            #         # exit(1)
            #         samples = np.concatenate((samples, np.expand_dims(tmp_v, axis=0)), 0)
            
            # samples[:10] *= 255
            # samples = torch.Tensor(np.moveaxis(samples, 3, 1))
            # # print(samples.shape)
            # samples = torch.reshape(samples, (5, 10, 3, 96, 96))
            # samples = torch.transpose(samples, 0, 1)
            # samples = torch.reshape(samples, (50, 3, 96, 96))

            # # fig = plt.figure()
            # plt.axis('off')
            # samples = make_grid(samples, nrow=5, padding=2, normalize=True,
            #             range=None, scale_each=False, pad_value=0)
            # save_image(samples, f'/vol/bitbucket/yw2621/SimCLR/figures/gradcam/z_{ds_name}_batch_{str(i)}.pdf')

def main_visualize_aug_gradcam(MODELS, num_imgs=10, forward_passes=20):
    ''' for each data augmentation, plot the gradcam when different intensities of data aug is applied.
    '''
    # get orginal dataset and sample images to be used
    test_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='test', download=True,
                                    transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256,
                                num_workers=12, drop_last=False, shuffle=False)

    tensors, _ = next(iter(test_loader))
    tensors = tensors[:num_imgs,:,:,:]
    tensor_to_pil = transforms.ToPILImage()

    gradcam_models = init_gradcam_models(MODELS)
    num_models = len(gradcam_models)
    # for each data aug, plot the gradual change of data augmemation, and the gradcam of differen models
    # the gradcam is plotted against data uncertainty.
    for aug_name in augs:
        t = augs[aug_name]
        print(aug_name)
        for j in range(num_imgs):
            samples = torch.empty((1,3,96,96))
            for i in range(10): # num of augmented images for each sing image
                trans = get_aug_loader(aug_name, t[i], return_loader=False)
                ddd = trans(tensor_to_pil(tensors[j])).unsqueeze(0)
                samples = torch.cat((samples, ddd), 0)
            samples = samples[1:]

            cams, predicts, ucts = get_gradcam_outputs(gradcam_models, samples, forward_passes)
            single_X = np.moveaxis(tensors[j].cpu().numpy().squeeze(), 0, 2)
            for model_index in range(len(gradcam_models)):
                for img_index in range(10):
                    tmp_v, _ = show_cam_on_image(single_X, cams[model_index][img_index], use_rgb=False)
                    tmp_v_tensor = torch.Tensor(np.expand_dims(np.moveaxis(tmp_v, 2, 0), axis=0)) / 255
                    samples = torch.cat((samples, tmp_v_tensor), 0)
            
            # print(samples.shape)
            samples = torch.reshape(samples, (5, 10, 3, 96, 96))
            samples = torch.transpose(samples, 0, 1)
            samples = torch.reshape(samples, (50, 3, 96, 96))
            samples = make_grid(samples, nrow=5, padding=4, normalize=False,
                            range=None, scale_each=False, pad_value=0)
            # show(samples)
            save_image(samples, f'/vol/bitbucket/yw2621/SimCLR/figures/gradcam_aug/{aug_name}_{str(j)}.pdf')


    
def main_plot_pgd_saliency(MODELS, args, num_ims=10):

    test_loader = get_aug_loader('no aug')
    X, y= next(iter(test_loader))
    X = X[:num_ims, : ,:, :].to(device)
    y = y[:num_ims].to(device)
    pgd_saliency_all_models(X, MODELS, args.fp, 0.13, 0.3, 40, name_prefix='pgd_')

    num_ims = num_ims//2
    for i in range(3):
        dataset = OODNoLabelDataset(i)
        test_loader = DataLoader(dataset, batch_size=256 * 2, num_workers=10, drop_last=False, shuffle=False)
        X, y= next(iter(test_loader))
        X = X[:num_ims, : ,:, :].to(device)
        y = y[:num_ims].to(device)
        pgd_saliency_all_models(X, MODELS, args.fp, 0.15, 0.4, 30, name_prefix=f'{dataset.ds_name}_pgd_')
        

def main_ood_uncertainty(model):
    ''' Get ood uncertainty for a single model.
    '''
    final_s = ''
    for i in range(3):
        dataset = OODNoLabelDataset(i)
        dataloader = DataLoader(dataset, batch_size=256 * 2,
                            num_workers=10, drop_last=False, shuffle=False)
        _, _, _, s= get_monte_carlo_predictions(dataset.ds_name,
                                dataloader, 
                                forward_passes=10, 
                                model=model, 
                                n_classes=10, 
                                n_samples=len(dataset))
        final_s = final_s + s + '\n'
    return final_s

def main_plot_roc(MODELS, args, dataset_type='ood'):
    ''' Plot roc with regarding to ood dataset or datset with a specific augmentation.

    the 'if' statement explictly determines the dataset_type.

    First, use get_group_uncertainty to get the uncertainty for the orginal dataset, 'origin_d'
    then use the local function to get uncertainty for other datasets and plot with plot_roc_1x3.

    In fact, for any two datasets, we can use get_group_uncertainty to get uncertainty and manually create the y_true, 
    then the roc can be plotted using plot_roc_3x3.
    '''
    in_d_loader = get_aug_loader('no aug')
    num_models = len(MODELS)

    def get_group_uncertainty(ds_name, dataloader):
        total_uncs, data_uncs, model_uncs = [], [], []
        for model_index in range(num_models):
            total_unc, data_unc, _, s= get_monte_carlo_predictions(ds_name,
                                        dataloader, 
                                        forward_passes=args.fp, 
                                        model=MODELS[model_index], 
                                        n_classes=10, n_samples=len(dataloader.dataset))
            total_uncs.append(total_unc)
            data_uncs.append(data_unc)
            model_uncs.append(total_unc - data_unc)
        return [total_uncs, data_uncs, model_uncs]
    
    origin_d = get_group_uncertainty('STL10', in_d_loader)

    y_trues = []
    new_datas = []
    if dataset_type == 'ood':
        for dataset_index in range(3):
            ood_dataset = OODNoLabelDataset(dataset_index)
            ood_loader = DataLoader(ood_dataset, batch_size=256 * 2,
                                    num_workers=10, drop_last=False, shuffle=False)
            ds_name =  ood_dataset.ds_name
            print(ds_name, len(ood_dataset))
            y_true = np.zeros((8000+len(ood_dataset), 1))
            y_true[8000:] = 1    
            ood_unc = get_group_uncertainty(ds_name, ood_loader)
            plot_single_roc(ood_unc, origin_d, y_true, ds_name)

    elif dataset_type == 'aug':
        for aug_name in augs.keys():
            index_map = {
            'gaussian_blur': 5,
            'crop_resize': 7,
            'gaussian_noise': 6,
            'color_jitter': 4,
            'salt_and_peper_noise': 4,
            'cutout': 6,
            }
            intensity_index = index_map[aug_name]
            aug_loader = get_aug_loader(aug_name, augs[aug_name][intensity_index])
            ds_name = 'stl10_'+ aug_name
            
            y_true = np.zeros((8000+len(aug_loader.dataset), 1))
            y_true[8000:] = 1     
            aug_unc = get_group_uncertainty(ds_name, aug_loader)
            plot_single_roc(aug_unc, origin_d, y_true, ds_name)


def main_aug_curve(MODELS, args):
    ''' For each data augmentation, plot how the accuarcy/ different uncertainties changes corresponding to the change in intensity of
    the data augmentation.

    This will plot a 2x2 plot, with acc and three three uncertainties.

    On each plot, there will be num_models=4 curves, representing the performance of each model
    '''
    for aug_name in augs.keys():
        plot_uncertainty_vs_one_aug(aug_name, MODELS, forward_passes=args.fp)
        print(aug_name, ' vs uncertainty plotted.\n\n\n')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('-dataset-name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('-data', metavar='DIR', default='../datasets',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50'],
                    help='model architecture')
    parser.add_argument('-dropout', default=0.1, type=float, help='initial dropout rate', dest='dropout')
    parser.add_argument('-fp', default=10, type=int, help='foward_passes', dest='fp')
    parser.add_argument('-ckpt', default='checkpoint_best.pth.tar',
                        help='dataset name')
    parser.add_argument('--limit', action='store_true',
                        help='Limit dropout to only before fc.')

    parser.add_argument('-mode', default=0, type=int, help='mode', dest='mode')
    args = parser.parse_args()
    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def plot_embed(embed, all_class_labels, lim, name = 'tsne' ):
    for i in range(4):
        fig = plt.figure()
        data = pd.DataFrame({'x0': list(embed[i*8000:(i+1)*8000, 0]),
                            'x1': list(embed[i*8000:(i+1)*8000, 1]),
                            'label': all_class_labels[i*8000:(i+1)*8000]}, )
        ax = sns.scatterplot(data=data, x='x0', y='x1', hue = 'label')
        ax.get_legend().remove()

        plt.subplots_adjust(top = 0.98, bottom = 0.02, right = 0.98, left = 0.02, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.set(ylabel=None)
        ax.set(xlabel=None)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(lim)
        plt.ylim(lim)
        plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/tsne/{name}_{i}.jpg')
        plt.close(fig)


def plot_tsne(MODELS, test_loader, ds_name, layer = 'final'):
    num_models = len(MODELS)
    all_class_labels = np.empty((0,))
    all_features = np.empty((0, 10)) if layer == 'final' else np.empty((0, 512))

    for model_index in range(num_models):
        if layer == 'final':
            tmp_model = MODELS[model_index]
        else:
            tmp_model = []
            for m in MODELS[model_index].backbone.children():
                tmp_model.append(m)
            tmp_model = tmp_model[:-1]
            tmp_model = nn.Sequential(*tmp_model)

        for j, (image, label) in enumerate(test_loader):
            image = image.to(device)
            all_class_labels = np.concatenate((all_class_labels, label.cpu().numpy())) 
            with torch.no_grad():
                conv_output = tmp_model(image)
            all_features = np.vstack((all_features, conv_output.cpu().numpy().squeeze()))

    print('plotting pca tsne.')
    if layer != 'final':
        pca = PCA(n_components=50)
        embed = pca.fit_transform(all_features)
        embed = TSNE(n_components=2, init='random').fit_transform(embed)
    else:
        embed = TSNE(n_components=2, init='random').fit_transform(all_features)
    plot_embed(embed, all_class_labels, [-90, 90], name = f'{ds_name}_{layer}_tsne')

    print('plotting pca.')
    pca = PCA(n_components=2)
    embed = pca.fit_transform(all_features)
    plot_embed(embed, all_class_labels, [-9, 9], name = f'{ds_name}_{layer}_pca')


def plot_tsne_all_dataset(MODELS):
    for o in range(3):
        dataset = OODNoLabelDataset(o)
        dataloader = DataLoader(dataset, batch_size=256 * 2,
                                num_workers=10, drop_last=False, shuffle=True)
        ds_name = dataloader.dataset.ds_name
        plot_tsne(MODELS, dataloader, ds_name, layer = 'final')
        plot_tsne(MODELS, dataloader, ds_name, layer = 'feature')
    
    plot_tsne(MODELS, get_aug_loader('no aug'), 'stl10', layer = 'final')
    plot_tsne(MODELS, get_aug_loader('no aug'), 'stl10', layer = 'feature')




def main():
    args = get_args()   
    args.fp = 5

    MODELS = load_four_models(args.arch, args.dropout)

    # plot_tsne_all_dataset(MODELS)

    # main_plot_roc(MODELS, args, 'ood')
    # main_plot_roc(MODELS, args, 'aug')

    # main_visualize_ood_gradcam(MODELS, num_imgs=10, forward_passes=args.fp)
    # main_visualize_aug_gradcam(MODELS, num_imgs=5, forward_passes=args.fp)

    main_plot_pgd_saliency(MODELS, args, 20)
    # test_loader = get_aug_loader('no aug')
    # for model in MODELS:
    #     t, d, acc, _= get_monte_carlo_predictions('aa',
    #                                 test_loader, 
    #                                 forward_passes=args.fp, 
    #                                 model=model,  # !
    #                                 n_classes=10, 
    #                                 n_samples=len(test_loader.dataset))

    # main_aug_curve(MODELS, args)
    # main_show_aug_effects()


if __name__ == "__main__":
    main()