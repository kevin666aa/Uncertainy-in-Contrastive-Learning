import os
import shutil
import torch
import yaml
import torch.nn as nn
from models.resnet_mcdropout import ResNetMCDropout
from models.resnet_simclr import ResNetSimCLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_a_model(limit, arch, dropout, ckpt):
    '''
    limit: if True, use ResNetSimCLR and only add one dropout layer,
    if False, use ResNetMCDropout directly
    '''
    ckpt = os.path.join('../outputs/', ckpt)
    if limit:
        model = ResNetSimCLR(arch, 10).to(device)
        model.backbone.avgpool = nn.Sequential(nn.Dropout(dropout), model.backbone.avgpool)
    else:
        model = ResNetMCDropout(arch, 10, dropoutrate=dropout).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    state_dict = checkpoint['state_dict']
    model = check_load_ckpt(model, state_dict)
    model.eval()
    enable_dropout(model)
    model.to(device)
    return model

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def check_load_ckpt(model, state_dict):
    mstat = model.state_dict().keys()
    k = 0
    for i in mstat:
        if i in state_dict:
            k+=1
    if k != len(state_dict):
        print(f'Warning. {k} out of {len(state_dict)} state dict loaded. Please check if the ckpt is loaded correctly.')
    
    model.load_state_dict(state_dict, strict=False)
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def get_log_dir(args, suffix=''):
    # n = '/vol/bitbucket/yw2621/outputs/' + args.arch + \
    #         '_lr' + str(len(str(args.lr).split('.')[1])) + \
    #         '_bs' + str(args.batch_size) + \
    #         '_' + args.aug
    n = '/vol/bitbucket/yw2621/outputs/' + 'CL_' + args.arch + \
            '_dropout' + str(args.dropout) + '_cutout'
    return n + suffix


def accuracy(output, target, topk=(1,), sum=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output: [[0.5, 0.2, 0.1, 0.1, 0.1]]
    # target: [1, 4]
    # topk (2,)
    # pred [0, 1, 2] -> [[0], [1], [2]]
    # pred
    # 0  3
    # 1  1
    # 2  0
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if sum:
                res.append(correct_k.mul_(100.0))
            else:
                res.append(correct_k.mul_(100.0 / batch_size))
        return res
