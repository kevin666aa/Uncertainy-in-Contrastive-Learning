# Uncertainy-in-Contrastive-Learning

Author: Yiran Wu          
Supervisor: Yingzhen Li

Individual Project for MSc in Artificial Intelligence &amp; Machine Learning at Imperial College London.

Despite we reserve the options to use Cifar10, we didn't test the program with this dataset and use STL10 throughout the project. We suggest to run with STL10 first and it is already set as default. The SimCLR implementation is adopted from https://github.com/sthalles/SimCLR.




We train four different models:

## 1. Vanilla CL model
- set dropout rate to 0 to avoid adding dropout layers.
- certain paths to datasets might need to be changed manually inside the code. (default path to save ckpt should be changed in func `get_log_dir` of `util.py` )
First train with SimCLR framework:

```
python run.py --epochs 500 --dropout 0
```
Then finetune the corresponding ckpt:
- if using `-limit`, this means we are not adding dropout layers.

```
finetune.py -limit -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]
```

## 2. CL+ dropout model
- set dropout rate to not equal to 0 to add dropout.

```
python run.py --epochs 500 --dropout 0.1
```

- then finetune:

```
python finetune.py -dropout 0.1 -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]
```


## 3. Vanilla MC dropout model
train a vanilla resnet-18 with dropout.

```
python mcdrop.py -ckpt [ckpt name] 
```

## 4. MC dropout model with extra data augmentations:

```
python mcdrop.py -ckpt [ckpt name] -use_aug
```

## 5. Inference time
- To plot figures, make sure you have the four models prepared.
- In `util_inference.py`, change the path to the four models in function `load_four_models`.
- In `inference.py`, uncomment corresponding 'main_' functions to run inference for different plots: [main_plot_roc, main_visualize_ood_gradcam, main_plot_pgd_saliency, main_aug_curve, plot_tsne_all_dataset, main_show_aug_effects]

```
python inference.py
```


