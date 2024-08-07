#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 13:46:01
@LastEditTime: 2019-04-15 09:23:44
"""
import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack

class PGD(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, config=None):
        """
        @description: Projected Gradient Descent (PGD)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(PGD, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(config)

    def _parse_params(self, config):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
        } 
        @return: None
        """
        self.eps = float(config.get("epsilon", 0.1))
        self.eps_iter = float(config.get("eps_iter", 0.01))
        self.num_steps = int(config.get("num_steps", 15))

    # 原始版
    def generate(self, xs=None, ys=None):
        """
        @description:
        @param {
            xs:
            ys:
        }
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted

        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        copy_xs = copy_xs + np.float32(
            np.random.uniform(-self.eps, self.eps, copy_xs.shape)
        )
    
        for _ in range(self.num_steps):
            var_xs = Variable(
                torch.from_numpy(copy_xs).float().to(device), requires_grad=True
            )
            var_ys = Variable(ys.to(device))

            outputs = self.model(var_xs)
            loss = self.criterion(outputs, var_ys)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            loss.backward()
    
            grad_sign = var_xs.grad.data.sign().cpu().numpy()
            copy_xs = copy_xs + self.eps_iter * grad_sign
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)
    
        adv_xs = torch.from_numpy(copy_xs)
    
        return adv_xs

    # mask 版
    # def generate(self, xs=None, ys=None):
    #     """
    #     @description:
    #     @param {
    #         xs:
    #         ys:
    #     }
    #     @return: adv_xs{numpy.ndarray}
    #     """
    #     device = self.device
    #     targeted = self.IsTargeted
    #
    #     copy_xs = np.copy(xs.numpy())
    #     xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
    #     # copy_xs = copy_xs + np.float32(
    #     #     np.random.uniform(-self.eps, self.eps, copy_xs.shape)
    #     # )
    #     mask = np.zeros(copy_xs.shape).astype(np.float32)
    #     mask[:,:,-16:,-16:] = 1.0
    #     # print(mask)
    #     for _ in range(self.num_steps):
    #         var_xs = Variable(
    #             torch.from_numpy(copy_xs).float().to(device), requires_grad=True
    #         )
    #         var_ys = Variable(ys.to(device))
    #
    #         outputs = self.model(var_xs)
    #         loss = self.criterion(outputs, var_ys)
    #         if targeted:
    #             loss = -self.criterion(outputs, var_ys)
    #         loss.backward()
    #
    #         grad_sign = var_xs.grad.data.sign().cpu().numpy()
    #         copy_xs = copy_xs + self.eps_iter * grad_sign * mask
    #         # copy_xs = copy_xs + self.eps_iter * grad_sign
    #         # print(copy_xs)
    #         copy_xs = np.clip(copy_xs, xs_min, xs_max)
    #         copy_xs = np.clip(copy_xs, 0.0, 1.0)
    #
    #     adv_xs = torch.from_numpy(copy_xs)
    #
    #     return adv_xs

    # def find_mask(self, grad_data):
    #     mask = np.zeros(grad_data.shape).astype(np.float32)
    #     for i in range(len(grad_data)):
    #         the_grad = grad_data[i]
    #         grad_abs = abs(the_grad[0]) + abs(the_grad[1]) + abs(the_grad[2])
    #         # print(grad_abs.shape)
    #         # print(grad_abs)

    #         size = 32
    #         for j in range(size):
    #             value = np.max(np.max(grad_abs, axis=0))
    #             pos = np.where(grad_abs == value)
    #             row = pos[0][0]
    #             col = pos[1][0]
    #             # print(row,col)
    #             grad_abs[row, col] = -1
    #             mask[i, :, row, col] = 1

    #     return mask

    # # 找梯度最大 然后 mask 版
    # def generate(self, xs=None, ys=None):
    #     """
    #     @description:
    #     @param {
    #         xs:
    #         ys:
    #     }
    #     @return: adv_xs{numpy.ndarray}
    #     """
    #     device = self.device
    #     targeted = self.IsTargeted

    #     copy_xs = np.copy(xs.numpy())
    #     xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
    #     # copy_xs = copy_xs + np.float32(
    #     #     np.random.uniform(-self.eps, self.eps, copy_xs.shape)
    #     # )

    #     # mask = np.zeros(copy_xs.shape).astype(np.float32)
    #     # mask[:,:,-16:,-16:] = 1.0
    #     # print(mask)

    #     for _ in range(self.num_steps):
    #         var_xs = Variable(
    #             torch.from_numpy(copy_xs).float().to(device), requires_grad=True
    #         )
    #         var_ys = Variable(ys.to(device))

    #         outputs = self.model(var_xs)
    #         loss = self.criterion(outputs, var_ys)
    #         if targeted:
    #             loss = -self.criterion(outputs, var_ys)
    #         loss.backward()

    #         grad_data = var_xs.grad.data.cpu().numpy()
    #         grad_sign = var_xs.grad.data.sign().cpu().numpy()
    #         mask = self.find_mask(grad_data)
    #         copy_xs = copy_xs + self.eps_iter * grad_sign * mask
    #         # copy_xs = copy_xs + self.eps_iter * grad_sign
    #         # print(copy_xs)
    #         copy_xs = np.clip(copy_xs, xs_min, xs_max)
    #         copy_xs = np.clip(copy_xs, 0.0, 1.0)

    #     adv_xs = torch.from_numpy(copy_xs)

    #     return adv_xs
