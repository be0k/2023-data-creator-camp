from torch import nn
from torchvision import models
import torch


def load_model(arch, device, length, PATH=None):
  if arch == 18:
    resnet = models.resnet18(pretrained=False)
  elif arch == 34:
    resnet = models.resnet34(pretrained=False)
  elif arch == 50:
    resnet = models.resnet50(pretrained=False)
  elif arch == 101:
    resnet = models.resnet101(pretrained=False)
  elif arch == 152:
    resnet = models.resnet152(pretrained=False)

  fc_in_features = resnet.fc.in_features
  resnet.fc = nn.Linear(fc_in_features,length)

  if PATH != None:
    resnet.load_state_dict(torch.load(PATH)[f'resnet{arch}'])

  resnet.to(device)
  return resnet


##########################################################################################

class EnsembleModel2(nn.Module):#mission2
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        #x3 = self.modelC(x)
        #x4 = self.modelD(x)
        #x5 = self.modelE(x)
        return x1 + x2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH_mission2 = './mission2.pt'

resnet18_2 = load_model(18, device, 42, PATH=PATH_mission2)
resnet34_2 = load_model(34, device, 42, PATH=PATH_mission2)
resnet50_2 = load_model(50, device, 42, PATH=PATH_mission2)
resnet101_2 = load_model(101, device, 42, PATH=PATH_mission2)
resnet152_2 = load_model(152, device, 42, PATH=PATH_mission2)

model = EnsembleModel2(resnet18_2, resnet34_2, resnet50_2, resnet101_2, resnet152_2)

##########################################################################################

class EnsembleModel3(nn.Module):#mission3
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE

    def forward(self, x):
        #x1 = self.modelA(x)
        #x2 = self.modelB(x)
        #x3 = self.modelC(x)
        #x4 = self.modelD(x)
        x5 = self.modelE(x)
        return x5


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH_mission3 = './mission3.pt'

resnet18_3 = load_model(18, device, 13, PATH=PATH_mission3)
resnet34_3 = load_model(34, device, 13, PATH=PATH_mission3)
resnet50_3 = load_model(50, device, 13, PATH=PATH_mission3)
resnet101_3 = load_model(101, device, 13, PATH=PATH_mission3)
resnet152_3 = load_model(152, device, 13, PATH=PATH_mission3)

model = EnsembleModel3(resnet18_3, resnet34_3, resnet50_3, resnet101_3, resnet152_3)
