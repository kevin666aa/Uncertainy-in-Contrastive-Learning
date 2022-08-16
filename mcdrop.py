from ast import arg
from unicodedata import name
import torch
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
from utils import save_checkpoint, accuracy
from models.resnet_mcdropout import ResNetMCDropout
from data_aug.gaussian_blur import GaussianBlur


def get_stl10_data_loaders(download, shuffle=False, batch_size=256, use_aug=True):
    s = 1
    size = 96

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    trans = [transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                # transforms.RandomApply([Cutout(1, 20)], p=0.9),
                transforms.ToTensor()]
    if not use_aug:
        print('not using aug.')
        trans = [transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()]    
    else:
        print('Using aug.')
    trans = transforms.Compose(trans)

    
    train_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='train', download=download,
                                  transform=trans)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    
    
    test_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def plot_acc(train_acc_list, test_acc_list, figname):
    train_acc_list, test_acc_list= np.round(np.array(train_acc_list), 2), np.round(np.array(test_acc_list), 2)
    x = np.arange(49, 251, 50)
    for i in x:
        plt.annotate('(%s, %s)' % (i, train_acc_list[i]), xy=(i, train_acc_list[i]), textcoords='data')
        plt.annotate('(%s, %s)' % (i, test_acc_list[i]), xy=(i, test_acc_list[i]), textcoords='data')
    plt.plot(train_acc_list, label = 'train')
    plt.plot(test_acc_list, label = 'test')
    plt.legend(loc="upper right")
    plt.savefig(figname)


def train(epochs, train_loader, test_loader, arch, ckpt_dir):
    dt = {'wd3': 0.008, 'wd4': 0.0008, 'wd5': 0.00008}
    ckpt_dir = os.path.join('../outputs/', ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        print(f'creating {ckpt_dir}')
        os.mkdir(ckpt_dir)
    model = ResNetMCDropout(arch, 10, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_acc_list, test_acc_list = [], []
    best_test_acc = 0
    for epoch in range(epochs):
        top1_train_accuracy = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        model.eval()
        enable_dropout(model)
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
    
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        train_acc_list.append(top1_train_accuracy.item())
        test_acc_list.append(top1_accuracy.item())
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 train accuracy: {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}")
        if top1_accuracy.item() > best_test_acc:
            best_test_acc = top1_accuracy.item()
            checkpoint_name = 'checkpoint_best.pth.tar'
            torch.save({
                'epoch': epoch,
                'arch': arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(ckpt_dir, checkpoint_name))

    plot_acc(train_acc_list, test_acc_list, os.path.join(ckpt_dir, 'acc.jpg'))
    print(f'best_test_acc : {best_test_acc}')

    del model
    del optimizer
    del criterion
    return top1_train_accuracy,  top1_accuracy, top5_accuracy

# https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('-dataset-name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('-data', metavar='DIR', default='../datasets',
                        help='path to dataset')
    parser.add_argument('-epochs', default=300, type=int, metavar='N',
                        help='train epochs')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50'],
                    help='model architecture')
    parser.add_argument('-ckpt', default='wd3',
                        help='path to ckpt dir')               
    args = parser.parse_args()

    train_loader, test_loader = get_stl10_data_loaders(download=True, use_aug=False)
    print("Dataset:", args.dataset_name, 'Datset dir', args.data)

    train(args.epochs, train_loader, test_loader, args.arch, args.ckpt)

if __name__ == "__main__":
    main()