#!/usr/bin/env python
# coding=UTF-8
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from .attack import Attack
import random
from scipy import stats as st
import torch.nn.functional as F

class MISMI(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, config=None):
        super(MISMI, self).__init__(model, device, IsTargeted)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._parse_params(config)
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
    
    def _parse_params(self, config):
        self.eps = float(config.get("epsilon", 0.1))
        self.eps_iter = float(config.get("eps_iter", 0.01))
        self.num_steps1 = int(config.get("num_steps1", 15))
        self.decay_factor = float(config.get("decay_factor", 1.0))
        self.num_steps2 = int(config.get("num_steps2", 10))
        self.posi = float(config.get("posi", 0.5))
        self.kernel_name = str(config.get("kernel_name", "linear"))
        self.len_kernel = int(config.get("len_kernel", 15))
        self.nsig = int(config.get("nsig", 3))
        self.resize_rate = float(config.get("resize_rate",0.9))
        self.diversity_prob = float(config.get("diversity_prob",0.5))
        self.random_start = bool(config.get("random_start",False))
    
    def generate(self, xs=None, ys=None):
        device = self.device
        targeted = self.IsTargeted
        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        
        var_xs = torch.tensor(
            copy_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_ys = torch.tensor(ys, device=device)

        if self.random_start:
            # Starting at a uniformly random point
            var_xs = var_xs + torch.empty_like(var_xs).uniform_(
                -self.eps, self.eps
            )
            var_xs = torch.clamp(var_xs, min=0, max=1).detach()
        stacked_kernel = self.stacked_kernel.to(self.device)
        
        momentum = 0  # 初始化动量为0
        for _ in range(self.num_steps1):
            # di
            var_xs = self.T(var_xs)
            # smi
            flag = True
            for i in range(self.num_steps2):
                var_xs = self.H(var_xs)
                
                outputs = self.model(var_xs)
                if targeted:
                    loss = -self.criterion(outputs, var_ys)
                else:
                    loss = self.criterion(outputs, var_ys)
                grad = torch.autograd.grad(
                    loss,var_xs,retain_graph=False,create_graph=False
                )[0]
                if flag:
                    all_grad = grad.data
                    flag = False
                else: 
                    all_grad += grad.data

            grad = all_grad / self.num_steps2
            # ti
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            # mi
            gradient = grad.data
            gradient_l1 = torch.norm(gradient, 1)
            gradient_l1 = gradient_l1.detach().cpu().numpy()
            gradient_np = gradient.detach().cpu().numpy()

            momentum = self.decay_factor * momentum + gradient_np / gradient_l1
            mask = self.find_mask(grad.data.cpu().numpy())
            copy_xs = copy_xs + self.eps_iter * np.sign(momentum) #* mask
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)
            var_xs = torch.tensor(
                copy_xs, dtype=torch.float, device=device, requires_grad=True
            )

        adv_xs = torch.from_numpy(copy_xs)
        return adv_xs   

    def T(self,image):    
        if random.random() < self.posi:
            mytransform1 = torchvision.transforms.RandomAffine(degrees=0,scale=(0.95,1),fillcolor=0,resample=False)
            image_transformed = mytransform1(image)
        else:
            image_transformed = image

        return image_transformed

    def H(self,image):   
        mytransform1 = torchvision.transforms.RandomAffine(translate=(0.01,0.01),degrees=1,scale=(0.97,1),fillcolor=0,resample=False)
        image_transformed = mytransform1(image)
        return image_transformed       
            
    def find_mask(self, grad_data):
        mask = np.zeros(grad_data.shape).astype(np.float32)
        for i in range(len(grad_data)):
            the_grad = grad_data[i]
            grad_abs = abs(the_grad[0]) + abs(the_grad[1]) + abs(the_grad[2])

            size = 64
            for j in range(size):
                value = np.max(np.max(grad_abs, axis=0))
                pos = np.where(grad_abs == value)
                row = pos[0][0]
                col = pos[1][0]
                # print(row,col)
                grad_abs[row, col] = -1
                mask[i, :, row, col] = 1
        return mask

    def kernel_generation(self):
        if self.kernel_name == "gaussian":
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == "linear":
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == "uniform":
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(
            np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
            / (kernlen + 1)
            * 2
        )
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x