import timm
import torch
import torch.nn as nn
from torchsummary import summary
device = torch.device('cuda')
class Convnext(nn.Module):
    def __init__(self, num_class1, num_class2, num_class3):
        super(Convnext, self).__init__()

        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.stages = model
        model_s = torch.nn.Sequential(*(list(model.children())[1]))

        self.backbone = torch.nn.Sequential(*(list(model.children())[0]), *(list(model_s.children())[:-2]))

        self.head_1 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))

        self.classifier1 = nn.Linear(768, num_class1, bias=True)

        self.head_2 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))
        self.classifier2 = nn.Linear(768, num_class2, bias=True)

        self.head_3 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))
        self.classifier3 = nn.Linear(768, num_class3, bias=True)
        
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.backbone(x)
        
        h1 = self.head_1(x)
        o1 = self.classifier1(h1)

        h2 = self.head_2(x)
        o2 = self.classifier2(h2)

        h3 = self.head_3(x)
        o3 = self.classifier3(h3)

        return o1, o2, o3

class Convnext_sharefeature(nn.Module):
    def __init__(self, num_class1, num_class2, num_class3):
        super(Convnext_sharefeature, self).__init__()

        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        
        model_s = torch.nn.Sequential(*(list(model.children())[1]))

        self.backbone = torch.nn.Sequential(*(list(model.children())[0]), *(list(model_s.children())[:-2]))

        self.head_1 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))
        self.classifier1 = nn.Linear(768, num_class1, bias=True)
  
        self.head_2 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))
        self.classifier2 = nn.Linear(768, num_class2, bias=True)

        self.head_3 = torch.nn.Sequential(*(list(model_s.children())[-2:]), (list(model.children())[-1]))
        self.classifier3 = nn.Linear(768, num_class3, bias=True)
        
        self.softmax_reg1 = nn.Linear(num_class1, num_class1)
        self.softmax_reg2 = nn.Linear(num_class1+num_class2, num_class2)
        self.softmax_reg3 = nn.Linear(num_class1+num_class2+num_class3, num_class3)

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)

        h1 = self.head_1(x)
        #o1 = self.classifier1(h1)

        h2 = self.head_2(x)
        #o2 = self.classifier2(h2)

        h3 = self.head_3(x)
        #o3 = self.classifier3(h3)

        level_1 = self.softmax_reg1(self.classifier1(h1))
        level_2 = self.softmax_reg2(torch.cat((level_1, self.classifier2(h2)), dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_1,level_2, self.classifier3(h3)), dim=1))# add layer3

        return level_1, level_2, level_3