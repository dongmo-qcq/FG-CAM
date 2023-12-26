import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.layers import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.hook = []

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def improve_resolution(self, I, target_layer):
        for i in range(len(self.features)-1, target_layer, -1):
            I = self.features[i].IR(I)
        return I
    
    def register_hook(self):
        for m in self.features:
            m.register_forward_hook(forward_hook)
    
    def remove_hook(self):
        for m in self.hook:
            m.remove()
        self.hook=[]

def create_features_modules(layers, batch_norm=False):
    features = []
    in_channels = 3
    for layer in layers:
        if layer == 'M':
            features += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, layer, kernel_size=3, padding=1)
            if batch_norm:
                features += [conv2d, BatchNorm2d(layer), ReLU(inplace=True)]
            else:
                features += [conv2d, ReLU(inplace=True)]
            in_channels = layer

    return nn.Sequential(*features)

models_param = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg11']))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False):
    model = VGG(create_features_modules(vgg11['A'], batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg13']))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg13'], batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg16']))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16_bn(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg16'], batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg19']))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False):
    model = VGG(create_features_modules(models_param['vgg19'], batch_norm=True))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
