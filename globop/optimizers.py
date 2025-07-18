import torch
import torch.optim as optim
from .base import SuGD


class AdaptiveOptimizer:
    def __init__(self, model, criterion, low_bound, up_bound, train_loader, device, optimizer_class, **optimizer_kwargs):
        self.model = model
        self.criterion = criterion
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.train_loader = train_loader
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None
        self.current_lr = None

    def step_epoch(self):
        batch = next(iter(self.train_loader))
        self.current_lr = SuGD(
            self.model, self.criterion, batch, self.low_bound, self.up_bound, self.device
        )
        print(f"  [AdaptiveOptimizer] Best LR: {self.current_lr:.5f}")
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.current_lr, **self.optimizer_kwargs
        )

    def step_batch(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_lr(self):
        return self.current_lr


# === Specific Wrappers ===

def SuperSGD(model, criterion, low_bound, up_bound, train_loader, device, **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, optim.SGD, **kwargs)

def SuperAdam(model, criterion, low_bound, up_bound, train_loader, device, **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, optim.Adam, **kwargs)

def SuperAdamW(model, criterion, low_bound, up_bound, train_loader, device, **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, optim.AdamW, **kwargs)

def SuperAdagrad(model, criterion, low_bound, up_bound, train_loader, device, **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, optim.Adagrad, **kwargs)

def SuperRMSprop(model, criterion, low_bound, up_bound, train_loader, device, **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, optim.RMSprop, **kwargs)
