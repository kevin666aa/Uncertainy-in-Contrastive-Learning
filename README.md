# Uncertainy-in-Contrastive-Learning
Individual Project for MSc in AI &amp; ML at Imperial College London

Despite we reserve the options to use Cifar10, we didn't test the program with this dataset and use STL10 throughout the project. We suggest to run with STL10 first and it is already set as default.

We train four different models.

## 1. Vanilla CL model
- set dropout rate to 0 to avoid adding dropout layers.
- certain paths to datasets might need to be changed manually inside the code. (default path to save ckpt should be changed in func get_log_dir of util.py )
- 
First train with SimCLR framework:

`python run.py --epochs 500 --dropout 0`
Then finetune the corresponding ckpt:
- if using '-limit', this means we are not adding dropout layers.

`python finetune.py -limit -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]`

## 2. CL+ dropout model
- set dropout rate to not equal to 0 to add dropout.

`python run.py --epochs 500 --dropout 0.1`

- then finetune:

`python finetune.py -dropout 0.1 -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]`


## 3. Vanilla MC dropout model
train a vanilla resnet-18 with dropout.

`python mcdrop.py -ckpt [ckpt name] `

## 4. MC dropout model with extra data augmentations:

`python mcdrop.py -ckpt [ckpt name] -use_aug`

## Inference time
In inference.py, uncomment corresponding 'main_' functions to run inference.
[main_plot_roc, main_visualize_ood_gradcam, main_plot_pgd_saliency, main_aug_curve, plot_tsne_all_dataset, main_show_aug_effects]

`python inference.py`
