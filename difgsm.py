# coding=UTF-8
import random
import torchvision
from torchvision import transforms
import cv2 as cv

import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class DIFGSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, config=None):
        """
        @description:  Diverse Inputs Iterative - Fast Gradient Sign Method (DIFGSM) 
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            IsTargeted:是否是目标攻击
            kwargs: 用户对攻击方法需要的参数
        } 
        @return: None
        """
        super(DIFGSM, self).__init__(model, device, IsTargeted)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._parse_params(config)


    def _parse_params(self, config):
        """
        @description: 
        @param {
            epsilon:沿着梯度方向步长的参数
        } 
        @return: None
        """
        self.eps = float(config.get("epsilon", 0.03))
        self.alpha = float(config.get("alpha", 0.01))
        self.num_steps = int(config.get("num_steps", 10))
        self.posi = float(config.get("posi", 0.5))

    def T(self,image):
        
        if random.random() < self.posi:
            mytransform1 = torchvision.transforms.RandomAffine(degrees=0,scale=(0.95,1),fillcolor=0,resample=False)
            image_transformed = mytransform1(image)
        else:
            image_transformed = image

        return image_transformed
    
   

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:原始的样本
            ys:样本的标签
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted
        copy_xs = np.copy(xs.numpy())
        var_ys = torch.tensor(ys, device=device)
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        for _ in range(self.num_steps):    
            var_xs = torch.tensor(
                copy_xs, dtype=torch.float, device=device, requires_grad=True
            )   
            var_xs = self.T(var_xs)
            outputs = self.model(var_xs)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            else:
                loss = self.criterion(outputs, var_ys)
            grad = torch.autograd.grad(
                loss,var_xs,retain_graph=False,create_graph=False
            )[0]
            grad_sign = grad.data.sign().cpu().numpy()
            
            delta = self.alpha * grad_sign
            copy_xs = copy_xs +delta
            copy_xs =np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)
        return adv_xs
