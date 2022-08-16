import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dropout_predictions(model, X, n_classes, forward_passes):
    softmax = nn.Softmax(dim=1)
    dropout_predictions = torch.empty((0, X.shape[0], n_classes)).to(device)
    for i in range(forward_passes):
        output = softmax(model(X)).unsqueeze(0)
        dropout_predictions = torch.vstack((dropout_predictions, output))
    return dropout_predictions

def get_ce_loss(model, X, y, n_classes, forward_passes):
    dropout_predictions = torch.empty((0, X.shape[0], n_classes)).to(device)
    for i in range(forward_passes):
        dropout_predictions = torch.vstack((dropout_predictions, model(X).unsqueeze(0)))
    mean = torch.mean(dropout_predictions, dim=0)
    return nn.CrossEntropyLoss()(mean, y)

def compute_all_uncertainty(dropout_predictions):
    """ shape(dropout_predictions): (forward_passes, n_samples, n_classes)
    """
    mean = torch.mean(dropout_predictions, dim=0) # shape (n_samples, n_classes)
    total_unc = -torch.sum(mean*torch.log(mean + sys.float_info.min), dim=-1) # shape (n_samples,)

    tmp = -torch.sum(dropout_predictions*torch.log(dropout_predictions + sys.float_info.min), dim=-1) # shape (forward_passes, n_samples)
    data_unc = torch.mean(tmp, dim=0) # shape (n_samples,)

    model_unc = total_unc - data_unc
    return total_unc, data_unc, model_unc

def plot_saliency_1x2(origin, saliency, name= '1'):
    ''' Plot 1x2 image matrxi with origin and saliency map from total uncertainty

    save in figs folder
    '''
    fig = plt.figure(figsize=(5, 2))
    rows, columns = 1, 2

    fig.add_subplot(rows, columns, 1)
    plt.axis('off')
    sns.heatmap(saliency, cmap='vlag')

    fig.add_subplot(rows, columns, 2)
    plt.axis('off')
    plt.imshow(origin)

    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/outputs/figs/' + name + '.jpg')

def plot_saliency_2x2(images, uncertainties, name= '1'):
    ''' Plot a 2x2 image matrix with 
    [orgin           , total_uncertainty]
    [data_uncertainty, model_uncertainty]
    
    images: list of images, len = 4 
    uncertainties: list of images, len = 3

    save in figs folder
    '''
    names = ['tota', 'data', 'model']
    fig = plt.figure(figsize=(4, 4))
    columns, rows = 2, 2
    for i, im in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        if i == 0:
            plt.title('image')
            plt.imshow(im)
        else:
            plt.title(names[i-1] + ': ' + str(round(uncertainties[i-1], 3)))
            plt.imshow(im, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/outputs/figs/' + name + '.jpg')
    plt.close(fig)

def plot_saliency_1x6(images, name= '1'):
    ''' Plot a 1x6 image matrix with 
    [orgin, total_uncertainty, data_uncertainty, 
    model_uncertainty_direct_backprop, model_uncertainty_subtraction, model_uncertainty_difference]
    
    images: list of images, len = 6
    uncertainties: list of images, len = 6
    '''
    names = ['image', 'total', 'data', 'model_dbp', 'model_sub', 'diff']
    columns, rows = 6, 1
    fig = plt.figure(figsize=(14, rows*2))
    for i, im in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title(names[i])
        if i == 0:
            plt.imshow(im, cmap=plt.cm.hot)
        else:
            sns.heatmap(im, cmap='vlag')
            
    #     highpass_3x3, highpass_5x5, gauss_highpass = high_pass_filter(im)
    #     fig.add_subplot(rows, columns, i+1 + 6)
    #     plt.axis('off')
    #     plt.title(names[i] + ' highpass_3x3')
    #     plt.imshow(highpass_3x3, cmap=plt.cm.hot)

    #     fig.add_subplot(rows, columns, i+1 + 12)
    #     plt.axis('off')
    #     plt.title(names[i] + ' highpass_5x5')
    #     plt.imshow(highpass_5x5, cmap=plt.cm.hot)

    #     fig.add_subplot(rows, columns, i+1 + 18)
    #     plt.axis('off')
    #     plt.title(names[i]+ ' gauss_highpass')
    #     plt.imshow(gauss_highpass, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/outputs/figs/' + name + '.jpg')
    plt.close(fig)

def plot_saliency_2x4(images, uct, name = '1'):
    ''' Plot a 2x4 image matrix with 
    [orgin, total_uncertainty, data_uncertainty, model_uncertainty_direct_backprop] + 
    [orgin, total_uncertainty-enhanced, data_uncertainty-enhanced, model_uncertainty_direct_backprop-enhanced] 
    
    images: list of images, (n, __)
    uncertainties: list of images,  (3, n, ___)
    '''
    names = ['image', 'total_{:2f}'.format(uct[0]), 'data_{:2f}'.format(uct[1]), 'model_dbp_{:2f}'.format(uct[2])] * 2
    images.append(images[0])
    for k in range(1, 4):
        top_20_thres = np.percentile(images[k], 90)
        # low_20_thres = np.percentile(images[k], 5)
        ma, mi = np.max(images[k]), np.min(images[k])
        tmp = np.where(images[k] > top_20_thres, ma, images[k])
        # tmp = np.where(tmp < low_20_thres, mi, tmp)
        images.append(tmp)
        
    rows, columns = 2,4
    fig = plt.figure(figsize=(10, rows*2))
    for i, im in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title(names[i])
        if i == 0 or i == 4:
            plt.imshow(im, cmap=plt.cm.hot)
        else:
            sns.heatmap(im, cmap='vlag')
            
    plt.tight_layout()
    plt.savefig('figs/' + name + '.jpg')
    plt.close(fig)

def plot_saliency_3xN(images, name = '1'):
    ''' Plot a 3xN image matrix with 
    N = 1 + num_models. 
    1 means that the original image will always be plotted as a saliency with regarding to 3xN images/maps. 
    
    See the variable 'titles' for detail.
    
    images: All images to be plotted, including saliency map.
    name: name of the saved figure.
    '''
    # for k in range(12):
    #     if k % 4 != 0:
    #         top_20_thres = np.percentile(images[k], 95)
    #         # low_20_thres = np.percentile(images[k], 5)
    #         ma, mi = np.max(images[k]), np.min(images[k])
    #         images[k] = np.where(images[k] > top_20_thres, ma, images[k])
    #         # tmp = np.where(tmp < low_20_thres, mi, tmp)

    model_names = {
        0 : 'dropout',
        1 : 'dropout+aug',
        2 : 'CL',
        3 : 'dropout+CL'}
    model_names = [model_names[k] for k in model_names.keys()]
    titles = ['total_uncertainty'] + model_names + ['data_uncertainty'] + model_names + ['model_uncertainty'] + model_names

    rows, columns = 3, (len(model_names)+1)
    fig = plt.figure(figsize=(columns*2.5, rows*2))
    for i, im in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title(titles[i])

        if i % (columns) == 0:
            plt.imshow(im, cmap=plt.cm.hot)
        else:
            sns.heatmap(im, cmap='rocket_r', vmin=0, vmax=0.05)
            
    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/SimCLR/figures/pgd/' + name + '.jpg')
    plt.close(fig)


def plot_single_saliency_3xN(images, name = '1'):
    ''' Plot a 3xN image matrix with 
    N = 1 + num_models. 
    1 means that the original image will always be plotted as a saliency with regarding to 3xN images/maps. 
    
    See the variable 'titles' for detail.
    
    images: All images to be plotted, including saliency map.
    name: name of the saved figure.
    '''
    transposed = []
    for i in range(5):
        transposed.append(images[i])
        transposed.append(images[i+5])
        transposed.append(images[i+10])

    plt.imshow(transposed[0])
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/pgd/im_{name}.png', transparent=True, bbox_inches="tight", pad_inches=0)

        

    transposed = transposed[3:]
    rows, columns =  4, 3
    fig = plt.figure(figsize=(columns*2.6, rows*2))
    for i, im in enumerate(transposed):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        
        ax = sns.heatmap(im, cmap='rocket_r', vmin=0, vmax=0.05)

    plt.tight_layout()
    plt.savefig('/vol/bitbucket/yw2621/SimCLR/figures/pgd/' + name + '.jpg')
    plt.close(fig)

def high_pass_filter(data):
    if len(data.shape) == 3:
        return data, data, data
    kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    highpass_3x3 = ndimage.convolve(data, kernel)

    # A slightly "wider", but sill very simple highpass filter 
    kernel = np.array([[-1, -1, -1, -1, -1],
                        [-1,  1,  2,  1, -1],
                        [-1,  2,  4,  2, -1],
                        [-1,  1,  2,  1, -1],
                        [-1, -1, -1, -1, -1]])
    highpass_5x5 = ndimage.convolve(data, kernel)

    # Another way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original. Here, we'll use a simple gaussian filter
    # to "blur" (i.e. a lowpass filter) the original.
    lowpass = ndimage.gaussian_filter(data, 3)
    gauss_highpass = data - lowpass
    return highpass_3x3, highpass_5x5, gauss_highpass


def pgd_saliency_total_uncertainty(X, model, forward_passes, n_classes, alpha=0.01, epsilon=0.2, num_iter=20):
    # https://adversarial-ml-tutorial.org/adversarial_examples/
    """ Construct FGSM adversarial examples on the examples X"""
    print('Computing pgd saliency')
    X.requires_grad = False
    num_im =  X.shape[0]
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        dropout_predictions = get_dropout_predictions(model, X+delta, n_classes, forward_passes) 
        total_unc, _, _ = compute_all_uncertainty(dropout_predictions)
        total_unc.sum().backward()
        # t = delta + num_im*alpha*delta.grad.data
        # print(torch.numel(t[t>epsilon]), torch.numel(t[t<-epsilon]))
        delta.data = (delta + num_im*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
        if t>10 and t % 20 == 0:
            saliency, _ = torch.max(delta.data, dim=1)
            saliency = saliency.cpu().numpy()
            for i in range(num_im):
                tmp = np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2)
                plot_saliency_1x2(saliency[i, :, :], tmp, 'pgd_'+ str(i) + '_ep_{:02e}'.format(epsilon) + '_it_' + str(t))

    # saliency, _ = torch.max(delta.data, dim=1)
    # saliency = saliency.cpu().numpy()
    # for i in range(num_im):
    #     tmp = np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2)
    #     plot_saliency_1x2(saliency[i, :, :], tmp, 'pgd_ep_'+ str(epsilon) + "_" + str(i))

    return delta.detach()


def pgd_saliency_ce_loss(X, y, model, forward_passes, n_classes, alpha=0.01, epsilon=0.2, num_iter=20):
    X.requires_grad = False
    num_im =  X.shape[0]
    delta = torch.zeros_like(X, requires_grad=True)
    for t in tqdm(range(num_iter)):
        loss = get_ce_loss(model, X+delta, y, n_classes, forward_passes)
        loss.backward()
        delta.data = (delta + num_im*alpha*delta.grad.data).clamp(-epsilon,epsilon)
    
    saliency_maps, _ = torch.max(delta.data, dim=1)
    saliency_maps= saliency_maps.cpu().numpy()

    for i in range(num_im):
        plot_saliency_1x2(np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2), saliency_maps[i], str(i))
    


def pgd_saliency_all_models(X, models, forward_passes, alpha, epsilon, num_iter, name_prefix='pgd_', n_classes=10):
    total_saliencies, data_saliencies, model_saliencies = [], [], []
    for model in tqdm(models):
        total_saliency, data_saliency, model_saliency = pgd_saliency_all_uncertainty(X, model, forward_passes, n_classes, alpha, epsilon, num_iter)
        # shape total_saliency:  (num_images, 96, 96)
        total_saliencies.append(total_saliency)  
        data_saliencies.append(data_saliency)
        model_saliencies.append(model_saliency)
    # shape total_saliencies: (num_models, num_images, 96, 96)
    
    
    for i in range(X.shape[0]):
        im = np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2)

        images = []
        for saliency in [total_saliencies, data_saliencies, model_saliencies]:
            images.append(im)
            images += [saliency[model_index][i] for model_index in range(len(models))]
        plot_single_saliency_3xN(images, name = name_prefix + str(i))
        #+ f'_aplha{alpha}_epsilon{epsilon}_iter{num_iter}'
    


def pgd_saliency_all_uncertainty(X, model, forward_passes, n_classes, alpha, epsilon, num_iter):
    def pgd_single_uncertainty(unct_index):
        ''' Plot pgd saliency map for one model with regarding to the three uncertainties: total, data and model uncertainty.
        '''
        # https://adversarial-ml-tutorial.org/adversarial_examples/
        """ Construct FGSM adversarial examples on the examples X"""
        # print('Computing pgd saliency')
        X.requires_grad = False
        num_im =  X.shape[0]
        delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            dropout_predictions = get_dropout_predictions(model, X+delta, n_classes, forward_passes) 
            ucts = compute_all_uncertainty(dropout_predictions)
            ucts[unct_index].sum().backward()
            delta.data = (delta + alpha*delta.grad.data).clamp(-epsilon,epsilon)
            delta.grad.zero_()

        saliency, _ = torch.max(delta.data, dim=1)
        saliency = abs(saliency.cpu().numpy())
        return saliency

    return [pgd_single_uncertainty(0),  pgd_single_uncertainty(1),  pgd_single_uncertainty(2)]


# def pgd_saliency_all_uncertainty(X, model, forward_passes, n_classes, alpha, epsilon, num_iter):
#     def pgd_single_uncertainty(unct_index):
#         ''' Plot pgd saliency map for one model with regarding to the three uncertainties: total, data and model uncertainty.
#         '''
#         # https://adversarial-ml-tutorial.org/adversarial_examples/
#         """ Construct FGSM adversarial examples on the examples X """
#         # print('Computing pgd saliency')
#         X.requires_grad = False
#         num_im =  X.shape[0]
#         max_delta = torch.zeros_like(X, requires_grad=False) # 5, 96, 96, 3
#         max_uct = torch.zeros(num_im).to(device) # 5
#         restarts = 2
#         for _ in range(restarts):
#             delta = torch.rand_like(X, requires_grad=True)
#             delta.data = delta.data * epsilon * 2 - epsilon

#             for t in range(num_iter):
#                 dropout_predictions = get_dropout_predictions(model, X+delta, n_classes, forward_passes) 
#                 ucts = compute_all_uncertainty(dropout_predictions)
#                 ucts[unct_index].sum().backward()
#                 delta.data = (delta + num_im*alpha*delta.grad.data).clamp(-epsilon,epsilon)
#                 delta.grad.zero_()
        
#             dropout_predictions = get_dropout_predictions(model, X+delta, n_classes, forward_passes) 
#             ucts = compute_all_uncertainty(dropout_predictions)

#             # print(ucts[unct_index].shape)
#             max_delta[ucts[unct_index] >= max_uct] = delta.detach()[ucts[unct_index] >= max_uct]
#             max_uct= torch.max(max_uct, ucts[unct_index])


#         saliency, _ = torch.max(delta.data, dim=1)
#         saliency = abs(saliency.cpu().numpy())
#         return saliency

#     return [pgd_single_uncertainty(0),  pgd_single_uncertainty(1),  pgd_single_uncertainty(2)]



def vanilla_saliency_all_uncertainty(X, model, forward_passes, n_classes):
    ''' All uncertainties are calculated and backpropagated separately.
    '''
    # model should be in eval mode, with dropout enabled.
    # https://github.com/sijoonlee/deep_learning/blob/master/cs231n/NetworkVisualization-PyTorch.ipynb
    print('Computing vanilla saliency.')
    num_im =  X.shape[0]
    X.requires_grad_()
    
    dropout_predictions = get_dropout_predictions(model, X, n_classes, forward_passes) # dropout predictions - shape (forward_passes, n_samples, n_classes)
    total_unc, data_unc, model_unc = compute_all_uncertainty(dropout_predictions)
    
    total_unc.sum().backward(retain_graph=True)
    total_saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    total_saliency = total_saliency.cpu().numpy()
    X.grad.zero_()

    data_unc.sum().backward(retain_graph=True)
    data_saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    data_saliency = data_saliency.cpu().numpy()
    X.grad.zero_()

    model_unc.sum().backward()
    model_saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    model_saliency = model_saliency.cpu().numpy()

    for i in range(num_im):
        tmp = np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2)
        plot_saliency_2x2([tmp,total_saliency[i, :, :], data_saliency[i, :, :], model_saliency[i, :, :]], 
                             [total_unc[i].item(), data_unc[i].item(), model_unc[i].item()],
                             'vn_' + str(i))
    return 

def vanilla_saliency_total_uncertainty(X, model, forward_passes, n_classes, num_im = 10):
    model.eval()
    enable_dropout(model)

    X = X[:num_im, : ,:, :].to(device)
    X.requires_grad_()
    
    softmax = nn.Softmax(dim=1)
    dropout_predictions = torch.empty((0, num_im, n_classes)).to(device)
    for i in tqdm(range(forward_passes)):
        output = softmax(model(X)).unsqueeze(0)
        dropout_predictions = torch.vstack((dropout_predictions, output))

    total_unc, _, _ = compute_all_uncertainty(dropout_predictions)
    total_unc.sum().backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    saliency = saliency.cpu().numpy()

    for i in range(num_im):
        tmp = np.moveaxis(X[i, :, :, :].detach().cpu().numpy(), 0, 2)
        plot_saliency_1x2(saliency[i, :, :], tmp, str(i))
    return saliency

