import torch
import torch.optim as optim
from .base import SuGD_Dynamic  # Make sure sugd_dynamic is defined in base.py or the correct module


class AdaptiveOptimizer:
    def __init__(self, model, criterion, train_loader, device, sugd_iters, w, output,
                 optimizer_class, alpha=0.9, initial_lr=1e-3, **optimizer_kwargs):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.sugd_iters = sugd_iters
        self.w = w
        self.output = output
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.alpha = alpha

        self.optimizer = optimizer_class(model.parameters(), lr=initial_lr, **optimizer_kwargs)
        self.current_lr = initial_lr

    def step_epoch(self):
        batch = next(iter(self.train_loader))
        self.current_lr = SuGD_Dynamic(
            self.model, self.criterion, batch, self.optimizer, self.device,
            alpha=self.alpha, sugd_iters=self.sugd_iters, w=self.w, output=self.output
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

def SuperSGD(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters=10, w = 0.2, output = "up", **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters, w, output, optim.SGD, **kwargs)

# def SuperAdam(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters=10, w = 0.2, output = "up", **kwargs):
#     return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters, w, output, optim.Adam, **kwargs)

def SuperAdam(model, criterion, train_loader, device, sugd_iters=10, w=0.2, output="up", alpha=0.9, initial_lr=1e-3, **kwargs):
    return AdaptiveOptimizer(model, criterion, train_loader, device, sugd_iters, w, output, optim.Adam, alpha=alpha, initial_lr=initial_lr, **kwargs)


def SuperAdamW(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters=10, w = 0.2, output = "up", **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters, w, output, optim.AdamW, **kwargs)

def SuperAdagrad(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters=10, w = 0.2, output = "up", **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters, w, output, optim.Adagrad, **kwargs)

def SuperRMSprop(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters=10, w = 0.2, output = "up", **kwargs):
    return AdaptiveOptimizer(model, criterion, low_bound, up_bound, train_loader, device, sugd_iters, w, output, optim.RMSprop, **kwargs)
