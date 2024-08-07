import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import stats as st

from .attack import Attack


class TIFGSM(Attack):
    def __init__(
        self,
        model=None,
        device=None,
        IsTargeted=None,
        config=None
    ):
        super(TIFGSM, self).__init__(model, device, IsTargeted)        
        self._parse_params(config)
        
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def _parse_params(self, config):
        """
        @description: 
        @param {
            eps (float): maximum perturbation. 
            alpha (float): step size. 
            steps (int): number of iterations. (Default: 10)
            decay (float): momentum factor. (Default: 0.0)
            kernel_name (str): kernel name. (Default: gaussian)
            len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
            nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
            resize_rate (float): resize factor used in input diversity. (Default: 0.9)
            diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
            random_start (bool): using random initialization of delta. (Default: False)

        } 
        @return: None
        """
        self.eps = float(config.get("epsilon", 0.03))
        self.alpha = float(config.get("alpha", 0.01))
        self.decay = float(config.get("decay",0.0))
        self.steps = int(config.get("num_steps", 10))
        self.kernel_name = str(config.get("kernel_name", "linear"))
        self.len_kernel = int(config.get("len_kernel", 15))
        self.nsig = int(config.get("nsig", 3))
        self.resize_rate = float(config.get("resize_rate",0.9))
        self.diversity_prob = float(config.get("diversity_prob",0.5))
        self.random_start = bool(config.get("random_start",False))

    def generate(self, images, targets):
        images = images.clone().detach().to(self.device)
        target_labels = targets.to(self.device)

        loss = torch.nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach().requires_grad_(True)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.input_diversity(adv_images))
            
            # Calculate loss
            cost = loss(outputs, target_labels)
            if self.IsTargeted:
                cost = -cost
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

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
    