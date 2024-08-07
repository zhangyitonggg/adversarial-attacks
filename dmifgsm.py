#!/usr/bin/env python
# coding=UTF-8
import numpy as np
import torch
from torch.autograd import Variable

import torchvision
import random
from .attack import Attack


class SMIFGSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, config=None):
        """
        @description: Enhancing Transferability of Adversarial Examples with Spatial Momentum (SMI-FGSM) 
        @param {
            model: 需要测试的模型
            device: 设备(GPU)
            IsTargeted: 是否是目标攻击
            kwargs: 用户对攻击方法需要的参数
        } 
        @return: None
        """
        super(SMIFGSM, self).__init__(model, device, IsTargeted)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._parse_params(config)

    def _parse_params(self, config):
        """
        @description: 
        @param {
            epsilon: 最大扰动范围
            eps_iter: 扰动步长
            num_steps: 迭代步数
            decay_factor: 衰减因子
        } 
        @return: None
        """
        self.eps = float(config.get("epsilon", 0.1))
        self.num_steps = int(config.get("num_steps", 15))
        self.decay_factor = float(config.get("decay_factor", 1.0))
        self.eps_iter = self.eps / self.num_steps
        self.posi = float(config.get("posi",0.5))

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
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        var_xs = torch.tensor(
            copy_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_ys = torch.tensor(ys, device=device)

        momentum = 0  

        for _ in range(self.num_steps):

            var_xs = self.T(var_xs)

            outputs = self.model(var_xs)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            else:
                loss = self.criterion(outputs, var_ys)

            grad = torch.autograd.grad(
                loss,var_xs,retain_graph=False,create_graph=False
            )[0]

            gradient = grad.data
            gradient_l1 = torch.norm(gradient, 1)
            gradient_l1 = gradient_l1.detach().cpu().numpy()
            gradient_np = gradient.detach().cpu().numpy()
            
            # 看到一个开源代码，其中momentum的更新与论文中不一致
            # momentum = self.decay_factor * momentum + grad_sign
            momentum = self.decay_factor * momentum + gradient_np / gradient_l1

            copy_xs = copy_xs + self.eps_iter * np.sign(momentum)
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)
            var_xs = torch.tensor(
                copy_xs, dtype=torch.float, device=device, requires_grad=True
            )
        
        adv_xs = torch.from_numpy(copy_xs)
        return adv_xs
