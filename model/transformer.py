import timm
import torch
import torch.nn as nn
from torchsummary import summary

model_Type ='vit_tiny_patch16_224'
class Transformer(nn.Module):
    def __init__(self, num_class1, num_class2, num_class3):
        super(Transformer, self).__init__()

        # 使用預訓練的 Vision Transformer 模型
        model = timm.create_model(model_Type, pretrained=True, num_classes=0)
        #model_0 = list(model.children())[-1]#model -
        
        
        #--------type 1 ---------#
        self.backbone = model
        ## 自定義分類器
        self.classifier1 = nn.Linear(self.backbone.num_features, num_class1, bias=True)
        self.classifier2 = nn.Linear(self.backbone.num_features, num_class2, bias=True)
        self.classifier3 = nn.Linear(self.backbone.num_features, num_class3, bias=True)
        a =0
        for name, param in self.backbone.named_parameters():
            #if name[7:9] < str(10):
            if a < 124:
                param.requires_grad = False
                a = a + 1       

    def forward(self, x):
        x = self.backbone(x)
        o1 = self.classifier1(x)
        o2 = self.classifier2(x)
        o3 = self.classifier3(x)
        return o1, o2, o3


class Transformer_sharefeature(nn.Module):
    def __init__(self, num_class1, num_class2, num_class3):
        super(Transformer_sharefeature, self).__init__()

        # 使用預訓練的 Vision Transformer 模型
        model = timm.create_model(model_Type, pretrained=True, num_classes=0)

        self.backbone = model
        ## 自定義分類器
        self.classifier1 = nn.Linear(self.backbone.num_features, num_class1, bias=True)
        self.classifier2 = nn.Linear(self.backbone.num_features, num_class2, bias=True)
        self.classifier3 = nn.Linear(self.backbone.num_features, num_class3, bias=True)
            
        self.softmax_reg1 = nn.Linear(num_class1, num_class1)
        self.softmax_reg2 = nn.Linear(num_class1+num_class2, num_class2)
        self.softmax_reg3 = nn.Linear(num_class1+num_class2+num_class3, num_class3)
        a =0
        for name, param in self.backbone.named_parameters():
            #if name[7:9] < str(10):
            if a < 124:
                param.requires_grad = False
                a = a + 1   
    def forward(self, x):
        x = self.backbone(x)

        h1 = self.classifier1(x)
        h2 = self.classifier2(x)
        h3 = self.classifier3(x)

        level_1 = self.softmax_reg1(h1)
        level_2 = self.softmax_reg2(torch.cat((level_1, h2), dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_1, level_2, h3), dim=1))  # add layer3

        return level_1, level_2, level_3