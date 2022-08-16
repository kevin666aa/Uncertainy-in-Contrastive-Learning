import torch.nn as nn
import torchvision.models as models



class ResNetMCDropout(nn.Module):

    def __init__(self, base_model, out_dim, dropoutrate=0.1):
        super(ResNetMCDropout, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim), # 4 layer
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model, dropoutrate)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name, dropoutrate):
        
        model = self.resnet_dict[model_name]   
        model.layer1 = update_dropout(model.layer1, dropoutrate) 
        model.layer2 = update_dropout(model.layer2, dropoutrate) 
        model.layer3 = update_dropout(model.layer3, dropoutrate) 
        model.layer4 = update_dropout(model.layer4, dropoutrate) 
        return model

    def forward(self, x):
        return self.backbone(x)

def update_dropout(net, dropout_rate=0.1):
    modules = []

    for i in net:
        modules.append(i)
        modules.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*modules)
