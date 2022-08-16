# Uncertainy-in-Contrastive-Learning
Individual Project for MSc in AI &amp; ML at Imperial College London

Despite we reserve the options to use Cifar10, we didn't test the program with this dataset and use STL10 throughout the project. We suggest to run with STL10 first and it is already set as default.

To train SimCLR models:


Vanilla CL model:
- set dropout rate to 0 to avoid adding dropout layers.
- certain paths to datasets might need to be changed manually inside the code. (default path to save ckpt should be changed in func get_log_dir of util.py )
- 

First train with SimCLR framework:
`python run.py --epochs 500 --dropout 0`
Then finetune the corresponding ckpt:
- if using '-limit', this means we are not adding dropout layers.
`python finetune.py -limit -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]`

CL+ dropout model:
- set dropout rate to not equal to 0 to add dropout.
`python run.py --epochs 500 --dropout 0.1`

- then finetune:
`python finetune.py -dropout 0.1 -ckpt [path to ckpt to be loaded] -saveckpt [ckpt to be save after finetuning]`
