import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['forward_hook','ReLU', 'BatchNorm2d', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'Conv2d']

def forward_hook(self, input, output):
    self.X = input[0].detach()
    self.X.requires_grad = True

def divide_with_zero(a, b):
    b_nozero = torch.where(b == 0, torch.ones_like(b), b)
    c = a / b_nozero
    return torch.where(b == 0, torch.zeros_like(b), c)

class ReLU(nn.ReLU):
    def IR(self, I):
        return I

class MaxPool2d(nn.MaxPool2d): 
    def IR(self, I):
        X = torch.clamp(self.X, min=0)
        Y = self.forward(X)
        I = divide_with_zero(I, Y)
        I = torch.autograd.grad(Y, X, I)[0] * X
        return I

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def IR(self, I):
        X = torch.clamp(self.X, min=0)
        Y = self.forward(X)
        I = divide_with_zero(I, Y)
        I = torch.autograd.grad(Y, X, I)[0] * X
        return I

class BatchNorm2d(nn.BatchNorm2d):
    def IR(self, I):
        return I

class Conv2d(nn.Conv2d):
    def IR(self, I):
        X = self.X
        positive_weight = torch.clamp(self.weight, min=0)
        nagative_weight = torch.clamp(self.weight, max=0)
        positive_input = torch.clamp(X, min=0)
        nagative_input = torch.clamp(X, max=0)
        if X.shape[1] == 3:
            B = X*0 + torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            H = X*0 + torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            
            Y1 = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding)
            Y2 = torch.conv2d(B, positive_weight, bias=None, stride=self.stride, padding=self.padding)
            Y3 = torch.conv2d(H, nagative_weight, bias=None, stride=self.stride, padding=self.padding)
            I = divide_with_zero(I, Y1 - Y2 - Y3)
            I = X * torch.autograd.grad(Y1, X, I)[0] - B * torch.autograd.grad(Y2, B, I)[0] - H * torch.autograd.grad(Y3, H, I)[0]
            
        else :
            Y1 = F.conv2d(positive_input, positive_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)
            Y2 = F.conv2d(nagative_input, nagative_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)
            I = divide_with_zero(I, Y1 + Y2)
            I = positive_input * torch.autograd.grad(Y1, positive_input, I)[0] + nagative_input * torch.autograd.grad(Y2, nagative_input, I)[0]
        
        return I
