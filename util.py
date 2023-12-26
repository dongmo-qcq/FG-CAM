import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from models.vgg import *
import cv2

def apply_transforms(image, size=224):

    if not isinstance(image, Image.Image):
        image = to_pil_image(image)

    transform = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

def get_model(model_name):
    if model_name == 'vgg11':
        model = vgg11(pretrained=True).cuda().eval()
    if model_name == 'vgg16':
        model = vgg16(pretrained=True).cuda().eval()
    if model_name == 'vgg13':
        model = vgg13(pretrained=True).cuda().eval()
    if model_name == 'vgg19':
        model = vgg19(pretrained=True).cuda().eval()
    if model_name == 'vgg11_bn':
        model = vgg11_bn(pretrained=True).cuda().eval()
    if model_name == 'vgg16_bn':
        model = vgg16_bn(pretrained=True).cuda().eval()
    if model_name == 'vgg13_bn':
        model = vgg13_bn(pretrained=True).cuda().eval()
    if model_name == 'vgg19_bn':
        model = vgg19_bn(pretrained=True).cuda().eval()
    
    return model

def get_target_layer(model,target_layer):
    if isinstance(model,VGG):
        return model.features[target_layer]

def visual_explanation(heatmap):
    heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
    heatmap = heatmap.detach().cpu().numpy()
    return cv2.resize(np.transpose(heatmap, (1, 2, 0)), (224, 224))