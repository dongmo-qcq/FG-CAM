import torch
from models.vgg import VGG
import torch.nn.functional as F

class FG_CAM:
    def __init__(self,model,base_cam):
        self.model = model
        self.base_cam = base_cam

    def svd(self,I):
        I = torch.nan_to_num(I[0])
        reshaped_I = (I).reshape(
                I.shape[0], -1)
        reshaped_I= reshaped_I - reshaped_I.mean(dim=1)[:,None]
        U, S, VT = torch.linalg.svd(reshaped_I, full_matrices=True)
        d = int(S.shape[0] * 0.1)
        s = torch.diag(S[:d],0)
        new_I = U[:,:d].mm(s).mm(VT[:d,:])
        new_I = new_I.reshape(I.size())
        return new_I

    def find_last_layer(self):
        if isinstance(self.model,VGG):
            return self.model.features[-1]
    
    def get_weight_by_grad_cam(self,input,target_class,layer):
        value = dict()
        def backward_hook(module, grad_input, grad_output):
            value["gradients"] = grad_output[0]
        def forward_hook(module, input, output):
            value["activations"] = output

        h1=layer.register_forward_hook(forward_hook)
        h2=layer.register_backward_hook(backward_hook)
        output = self.model(input)
        output[0][target_class].backward()
        h1.remove()
        h2.remove()
        return value["gradients"],value["activations"]
    
    def get_weight_by_score_cam(self,input,target_class,layer):
        value = dict()
        def forward_hook(module, input, output):
            value["activations"] = output
        h=layer.register_forward_hook(forward_hook)

        with torch.no_grad():
            self.model(input)
            h.remove()
            activations = value["activations"]
            weight = None
            batch = 8
            saliency_map = F.interpolate(activations, size=(224, 224), mode='bilinear', align_corners=False)
            saliency_map = torch.nan_to_num(saliency_map)
            maxs = saliency_map.view(saliency_map.size(0),
                                        saliency_map.size(1), -1).max(dim=-1)[0]
            mins = saliency_map.view(saliency_map.size(0),
                                        saliency_map.size(1), -1).min(dim=-1)[0]
            eps = torch.where(maxs==0,1e-9,0.0)
            saliency_map = (saliency_map - mins[:,:,None,None])/(maxs[:,:,None,None]-mins[:,:,None,None]+eps[:,:,None,None])
            saliency_map = saliency_map[0]

            for i in range(0,saliency_map.size(0),batch):
                x = input * saliency_map[i:i+batch,None,:,:]
                output =self.model(x)
                output = torch.softmax(output,dim=1)
                y = output[:,target_class]
                if i==0:
                    weight = y.clone()
                else:
                    weight = torch.cat([weight,y])
            return weight,activations


    def get_explanation_component(self,input,target_class,layer=None):
        if layer is None:
            layer = self.find_last_layer()
        if self.base_cam.lower() == 'grad_cam':
            weight,activation = self.get_weight_by_grad_cam(input,target_class,layer)
            I = torch.mean(weight,dim=(2,3),keepdim=True)*activation
        if self.base_cam.lower() == 'score_cam':
            weight,activation = self.get_weight_by_score_cam(input,target_class,layer)
            I = weight[None,:,None,None]*activation

        return I

    
    def forward(self, input, denoising, target_layer, target_class):
        if target_class is None:
            output = self.model(input)
            target_class = output.argmax(dim=-1)[0].item()

        I=self.get_explanation_component(input,target_class)
        self.model.register_hook()
        self.model(input)
        self.model.remove_hook()
        if denoising:
            I = self.svd(I)
        I = self.model.improve_resolution(I,target_layer)
        I = torch.sum(I,dim=1)
        return I,target_class

    def __call__(self,input,denoising,target_layer, target_class=None):
        return self.forward(input,denoising,target_layer,target_class)