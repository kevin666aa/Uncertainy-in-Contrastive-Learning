from ast import arg
from unicodedata import name
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import logging
import yaml
import pandas as pd
import torch.nn as nn

from models.resnet_mcdropout import ResNetMCDropout, update_dropout
from models.resnet_simclr import ResNetSimCLR
from utils import accuracy, check_load_ckpt, enable_dropout

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='train', download=download,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)


    test_dataset = datasets.STL10('/vol/bitbucket/yw2621/datasets', split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('/vol/bitbucket/yw2621/datasets', train=True, download=download,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.CIFAR10('/vol/bitbucket/yw2621/datasets', train=False, download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def train(model, epochs, train_loader, test_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
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
        with torch.no_grad():
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
            
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
    
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        # logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 train accuracy: {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}")
        if top1_accuracy.item() > best_test_acc:
            best_test_acc = top1_accuracy.item()
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(OUTPUTDIR, args.saveckpt))


    return model


def get_model(args):
    state_dict = torch.load(os.path.join(OUTPUTDIR, args.ckpt), map_location=device)['state_dict']
    # remove prefix
    for k in list(state_dict.keys()):
        if k.startswith('backbone.fc'):
            del state_dict[k]
    
    # get model 
    if args.limit:
        print('Adding only 1 dropout before fc.')
        model = ResNetSimCLR(args.arch, 10).to(device)
        model.backbone.avgpool = nn.Sequential(nn.Dropout(0.1), model.backbone.avgpool)
    else:
        model = ResNetMCDropout(args.arch, 10, dropoutrate=args.dropout).to(device)
    
    # load ckpt
    model = check_load_ckpt(model, state_dict)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if not 'fc' in name:
            param.requires_grad = False
            
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Num of params to be trained:', len(parameters))
    # assert len(parameters) == 2  # fc.weight, fc.bias
    return model





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
OUTPUTDIR = '/vol/bitbucket/yw2621/outputs'
CSVPATH = '/vol/bitbucket/yw2621/results.csv'

def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('-dataset-name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('-data', metavar='DIR', default='../datasets',
                        help='path to dataset')
    parser.add_argument('-ckpt', default='ckpt_best', type=str, metavar='N',
                        help='pretrain ckpt epochs')
    parser.add_argument('-epochs', default=150, type=int, metavar='N',
                        help='train epochs')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
    parser.add_argument('--limit', action='store_true',
                        help='Limit dropout to only before fc.')
    parser.add_argument('--dropout', default=0, type=float, help='initial dropout rate', dest='dropout')
    parser.add_argument('-saveckpt', type=str, 
                        help='where to save the finetuned ckpt.')

    args = parser.parse_args()

    if args.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=True)
    elif args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=True)
    print("Dataset:", args.dataset_name, 'Datset dir', args.data)
    
    model = get_model(args)
    model = train(model, args.epochs, train_loader, test_loader, args)
    return




if __name__ == "__main__":
    main()







#
    # for k in list(state_dict.keys()):
    #     if k.startswith('backbone.'):
    #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
    #             state_dict[k[len("backbone."):]] = state_dict[k]
    #     del state_dict[k]
    # if arch == 'resnet18':
    #     model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    # elif arch == 'resnet50':
    #     model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
    # load model


        # if args.dropout != 0:
    #     model.layer1 = update_dropout(model.layer1, args.dropout) 
    #     model.layer2 = update_dropout(model.layer2, args.dropout) 
    #     model.layer3 = update_dropout(model.layer3, args.dropout) 
    #     model.layer4 = update_dropout(model.layer4, args.dropout) 


        # logfile = os.path.join(OUTPUTDIR, args.saveckpt + '.log')
    # if not os.path.exists(logfile):
    #     with open(logfile, 'w'): pass
    # logging.basicConfig(filename=logfile, level=logging.DEBUG)

        #     if len(k) > 16 and k[16] == '1':
    #         print(k)
    #         tmp = k[:16] +  '2' + k[17:]
    #         state_dict[tmp] = state_dict[k]
    #         del state_dict[k]