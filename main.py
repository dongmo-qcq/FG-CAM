import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import util
from FG_CAM import FG_CAM
from PIL import Image
from image_net import ImageNet

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vgg16_bn',
                        help='Implemented model,such as vgg16, vgg16_bn')
    parser.add_argument('--base_cam', type=str, default="grad_cam",
                        help='base cam, grad_cam or score_cam')
    parser.add_argument('--denoising', type=bool, default=False,
                        help='Whether to use FG-CAM with denoising or not')
    parser.add_argument('--target_layer', type=int, default=13,
                        help='target_layer, -1 represents the input layer')
    parser.add_argument('--target_class', type=int, default=None,
                        help='target_class, default model\'s predicted class')
    return parser

def main(opts):    
    model = util.get_model(opts.model)
    fg_cam = FG_CAM(model,opts.base_cam)
    image_net = ImageNet()
    paths = os.listdir('./images')
    for path in paths:
        img_path = './images/{}'.format(path)
        input_image = Image.open(img_path).convert('RGB')
        image = np.array(input_image)
        image = cv2.resize(image,(224,224))
        input = util.apply_transforms(input_image).cuda()
        
        explanation,target_class = fg_cam(input,opts.denoising,opts.target_layer,opts.target_class)
        explanation = torch.relu(explanation)
        explanation = util.visual_explanation(explanation)
        print(path,'        ',image_net.get_class_name(target_class))
        save_path = './results/{}-{}-{}'.format("fg_"+opts.base_cam,image_net.get_class_name(target_class), img_path.split('/')[-1])

        plt.figure(figsize=(10, 4))
        c = 2
        if opts.target_layer!=-1:
            c = 3

        plt.subplot(1, c , 1)
        plt.imshow(image)
        plt.title('input')
        plt.axis('off')

        plt.subplot(1, c, 2)
        plt.imshow(explanation)
        plt.title('fg_'+opts.base_cam)
        plt.axis('off')

        if c==3:
            layer = util.get_target_layer(model,opts.target_layer)
            base_cam_explanation = fg_cam.get_explanation_component(input,target_class,layer)
            base_cam_explanation = torch.sum(base_cam_explanation,dim=1)
            base_cam_explanation = torch.relu(base_cam_explanation)
            base_cam_explanation = util.visual_explanation(base_cam_explanation)
            plt.subplot(1, c, 3)
            plt.imshow(base_cam_explanation)
            plt.title(opts.base_cam)
            plt.axis('off')

        plt.tight_layout()
        plt.draw()
        plt.savefig(save_path)
        plt.clf()
        plt.close()
        
    print("Done")

if __name__ == '__main__':
    opts = get_argparser().parse_args()
    main(opts)
