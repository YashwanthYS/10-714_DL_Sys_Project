"""Optimization module"""
import needle as ndl
from needle import ops
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            if self.weight_decay != 0.0:
                g = g + p * self.weight_decay

            if self.momentum != 0.0:
                if p not in self.u:
                    self.u[p] = (p * 0.0).detach()
                self.u[p] = (self.u[p] * self.momentum + g * (1.0 - self.momentum)).detach()
                upd = self.u[p]
            else:
                upd = g

            new_p = (p - upd * self.lr).detach()
            if new_p.dtype != p.dtype or new_p.device != p.device:
                new_p = ndl.Tensor(new_p.cached_data, device=p.device, dtype=p.dtype)
            p.data = new_p.data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)
                self.v[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)

            self.m[p] = (self.beta1 * self.m[p].data + (1.0 - self.beta1) * grad).data

            self.v[p] = (self.beta2 * self.v[p].data + (1.0 - self.beta2) * grad * grad).data

            bias_correction1 = 1.0 - self.beta1 ** self.t
            bias_correction2 = 1.0 - self.beta2 ** self.t
            
            m_hat = self.m[p].data / bias_correction1
            v_hat = self.v[p].data / bias_correction2

            update = (self.lr * m_hat / (v_hat ** 0.5 + self.eps)).realize_cached_data()
            p.cached_data = p.realize_cached_data() - update
        ### END YOUR SOLUTION
